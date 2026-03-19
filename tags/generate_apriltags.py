#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow",
#   "opencv-contrib-python",
#   "numpy",
# ]
# ///
"""
AprilTag Generator and Tiler for 3D Reconstruction

Generates all 35 Tag36h11 AprilTags (IDs 0–34) at specified physical sizes,
tiles them across A4 pages with crosshair registration marks and vertical
overlap, exports tag_configs.csv for COLMAP metadata, and saves a PDF of all
1×1 single-page tags.

Tag bit patterns from the official OpenCV DICT_APRILTAG_36h11 dictionary.

Usage:
    uv run generate_apriltags.py

Tiling model
------------
Horizontal: tag fills exactly N full A4-width columns (no horizontal margin so
            columns = round(target / 210mm) exactly).  Assembly: butt adjacent
            pages edge-to-edge; fold the last 10 mm of each page behind its
            neighbour for the physical gluing flap.

Vertical:   10 mm content overlap at every row seam (adjacent rows share 10 mm
            of tag pixels) and 10 mm top/bottom margins for crosshair
            registration marks.  This is where all alignment helpers live.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Tag36h11 geometry: 6×6 data grid inside a 1-px black border + 1-px white
# quiet zone → total raw image = 10×10 pixels.
DATA_SIDE  = 6
TOTAL_SIDE = 10

# ---------------------------------------------------------------------------
# Physical / print configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("apriltag_output")
CSV_PATH = OUTPUT_DIR / "tag_configs.csv"

DPI = 300
DPMM = DPI / 25.4                   # ≈ 11.811 dots-per-mm

# A4 page (portrait) in pixels at DPI
A4_W_PX = round(210.0 * DPMM)      # 2480
A4_H_PX = round(297.0 * DPMM)      # 3508

# Vertical tiling parameters (all in mm → converted below)
MARGIN_MM  = 10.0   # top/bottom margin per page (holds crosshairs + label)
OVERLAP_MM = 10.0   # vertical content overlap between adjacent rows

MARGIN_PX  = round(MARGIN_MM  * DPMM)   # 118
OVERLAP_PX = round(OVERLAP_MM * DPMM)   # 118

# Usable content height per A4 page (inside top/bottom margins)
USABLE_H_PX = A4_H_PX - 2 * MARGIN_PX                # 3272
# Step between row-tile origins (unique content added per row)
STEP_H_PX   = USABLE_H_PX - OVERLAP_PX                # 3154

CROSSHAIR_MM = 5.0                  # arm length of registration crosshair
CROSSHAIR_PX = round(CROSSHAIR_MM * DPMM)

# Generation spec: (first_id, last_id_inclusive, grid_n)
# grid_n  →  n×n tiles  →  tag physical width = n × 210mm = n × A4_W_PX
TAG_SPECS = [
    (0,   9,  1),   # 1×1  → 210mm tag
    (10, 13,  2),   # 2×2  → 420mm tag
    (14, 14,  4),   # 4×4  → 840mm tag
    (15, 34,  1),   # 1×1  → 210mm tag
]

# ---------------------------------------------------------------------------
# Bit-pattern rendering — uses OpenCV's official DICT_APRILTAG_36h11 to get
# canonical bit patterns, then renders with a proper 10×10 structure including
# the white quiet zone that the apriltag3 detector requires.
# The previous custom spiral renderer produced undetectable tags.
# ---------------------------------------------------------------------------

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def render_tag(tag_id: int, target_px: int) -> Image.Image:
    """
    Return a target_px × target_px grayscale PIL Image of the official
    Tag36h11 pattern for tag_id.

    Samples the 6×6 data bits from OpenCV's canonical marker, then builds
    the full 10×10 structure (white outer ring + black border + data cells)
    before scaling up — identical to generate_more_apriltags.py's approach.
    """
    # Sample at 800×800 (100px per cell) to read data bits cleanly
    cell = 100
    gen = np.zeros((8 * cell, 8 * cell), dtype=np.uint8)
    cv2.aruco.generateImageMarker(_ARUCO_DICT, tag_id, 8 * cell, gen, 1)

    bits = np.array([
        [1 if gen[cell + r * cell + cell // 2, cell + c * cell + cell // 2] > 127 else 0
         for c in range(DATA_SIDE)]
        for r in range(DATA_SIDE)
    ], dtype=np.uint8)

    # Build 10×10 PIL image with white outer ring (quiet zone) + black border
    pixels = [[255] * TOTAL_SIDE for _ in range(TOTAL_SIDE)]
    for i in range(1, 9):
        pixels[1][i] = 0;  pixels[8][i] = 0
        pixels[i][1] = 0;  pixels[i][8] = 0
    for dr in range(DATA_SIDE):
        for dc in range(DATA_SIDE):
            pixels[dr + 2][dc + 2] = 255 if bits[dr][dc] else 0
    flat = bytes(pixels[r][c] for r in range(TOTAL_SIDE) for c in range(TOTAL_SIDE))
    raw = Image.frombytes("L", (TOTAL_SIDE, TOTAL_SIDE), flat)
    return raw.resize((target_px, target_px), Image.NEAREST)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _font(size_mm: float = 3.0) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size_px = round(size_mm * DPMM)
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _crosshair(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    arm = CROSSHAIR_PX
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=0, width=2)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=0, width=2)
    r = max(2, arm // 5)
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=0, width=2)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_tag(
    tag_img: Image.Image,
    tag_id: int,
    grid_n: int,
    out_dir: Path,
) -> None:
    """
    Tile *tag_img* (grid_n × A4_W_PX square) onto A4 pages and save each
    as a high-resolution PNG.

    Horizontal: each column covers exactly A4_W_PX pixels of the tag (no
    horizontal margin, no content overlap).  grid_n columns total.

    Vertical: rows slide by STEP_H_PX with OVERLAP_PX content overlap at each
    row seam; 10 mm top/bottom margins hold crosshairs and labels.
    """
    tag_px = tag_img.width

    # How many row-tiles do we need?
    if tag_px <= USABLE_H_PX:
        n_rows = 1
    else:
        n_rows = math.ceil((tag_px - USABLE_H_PX) / STEP_H_PX) + 1

    n_cols = grid_n   # exact by construction

    font = _font(3.0)

    for row in range(n_rows):
        for col in range(n_cols):
            # ---- Horizontal slice (full A4 width, no overlap) ----
            src_x1 = col * A4_W_PX
            src_x2 = src_x1 + A4_W_PX          # always A4_W_PX wide

            # ---- Vertical slice (with overlap at seams) ----
            src_y1 = row * STEP_H_PX
            src_y2 = src_y1 + USABLE_H_PX       # USABLE_H_PX tall (overlaps next row)
            # Clamp to tag boundary
            src_y2 = min(src_y2, tag_px)

            crop = tag_img.crop((src_x1, src_y1, src_x2, src_y2))
            crop_h = crop.height

            # ---- Paste onto blank A4 canvas ----
            canvas = Image.new("RGB", (A4_W_PX, A4_H_PX), color=(255, 255, 255))
            # Tag content starts at top margin
            canvas.paste(crop.convert("RGB"), (0, MARGIN_PX))

            # ---- Crosshairs ----
            draw = ImageDraw.Draw(canvas)

            # Four corners of the content area (inside margin boundaries)
            content_top    = MARGIN_PX
            content_bottom = MARGIN_PX + crop_h
            content_left   = 0
            content_right  = A4_W_PX

            # Top-left and top-right (always)
            _crosshair(draw, content_left  + CROSSHAIR_PX + 4, content_top)
            _crosshair(draw, content_right - CROSSHAIR_PX - 4, content_top)
            # Bottom-left and bottom-right (always)
            _crosshair(draw, content_left  + CROSSHAIR_PX + 4, content_bottom)
            _crosshair(draw, content_right - CROSSHAIR_PX - 4, content_bottom)

            # Vertical seam tick marks: at col boundaries in the top/bottom margin
            # These help align pages horizontally when taping columns together
            for tick_col in range(1, n_cols):
                tx = tick_col * A4_W_PX        # seam x in assembled space
                # tx is on THIS page only when col == tick_col-1 (right edge) or tick_col (left edge)
                local_tx = tx - col * A4_W_PX  # in page coordinates
                if 0 <= local_tx <= A4_W_PX:
                    # top margin tick
                    draw.line([(local_tx, 0), (local_tx, MARGIN_PX - 4)], fill=(150, 150, 150), width=2)
                    # bottom margin tick
                    draw.line([(local_tx, content_bottom + 4), (local_tx, A4_H_PX)], fill=(150, 150, 150), width=2)

            # ---- Label ----
            label = (
                f"Tag {tag_id}  |  "
                f"Page col {col + 1}/{n_cols}, row {row + 1}/{n_rows}  |  "
                f"V-overlap: {OVERLAP_MM:.0f}mm"
            )
            draw.text((CROSSHAIR_PX * 2 + 8, 4), label, fill=(80, 80, 80), font=font)

            fname = f"tag{tag_id:03d}_r{row}c{col}.png"
            canvas.save(out_dir / fname, dpi=(DPI, DPI))
            print(f"    → {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate geometry once
    print(f"A4 page: {A4_W_PX}×{A4_H_PX} px at {DPI} DPI")
    print(f"Vertical margin: {MARGIN_MM:.0f}mm ({MARGIN_PX}px)  |  "
          f"V-overlap: {OVERLAP_MM:.0f}mm ({OVERLAP_PX}px)  |  "
          f"V-step: {STEP_H_PX}px ({STEP_H_PX/DPMM:.1f}mm)")
    print(f"Usable content height per page: {USABLE_H_PX}px ({USABLE_H_PX/DPMM:.1f}mm)")
    print()

    csv_rows: list[dict] = []
    single_page_tiles: list[Path] = []   # collect 1×1 tiles for PDF

    for start_id, end_id, grid_n in TAG_SPECS:
        tag_px = grid_n * A4_W_PX          # exact integer, no rounding issues
        actual_mm = tag_px / DPMM
        actual_m  = tag_px / DPMM / 1000.0

        # Verify tile counts before generating
        n_rows = (math.ceil((tag_px - USABLE_H_PX) / STEP_H_PX) + 1
                  if tag_px > USABLE_H_PX else 1)
        n_cols = grid_n

        print(
            f"=== Tag36h11  IDs {start_id}–{end_id}  "
            f"{grid_n}×{grid_n} grid  "
            f"tag={tag_px}px ({actual_mm:.2f}mm)  "
            f"→ {n_cols} col × {n_rows} row tiles ==="
        )

        tag_dir = OUTPUT_DIR / f"tags_{grid_n}x{grid_n}"
        tag_dir.mkdir(parents=True, exist_ok=True)

        for tag_id in range(start_id, end_id + 1):
            print(f"  Tag {tag_id}:")

            big = render_tag(tag_id, tag_px)

            # Full-size composite (reference / QC)
            composite = tag_dir / f"tag{tag_id:03d}_full.png"
            big.save(composite, dpi=(DPI, DPI))
            print(f"    → {composite.name}  (full composite, {tag_px}px)")

            tile_tag(big, tag_id, grid_n, tag_dir)

            if grid_n == 1:
                single_page_tiles.append(tag_dir / f"tag{tag_id:03d}_r0c0.png")

            csv_rows.append({
                "Tag_ID": tag_id,
                "Actual_Size_In_Meters": round(actual_m, 6),
            })

    # COLMAP metadata
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Tag_ID", "Actual_Size_In_Meters"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # PDF of all single-page (1×1) tags
    if single_page_tiles:
        pages = [Image.open(p) for p in single_page_tiles]
        pdf_path = OUTPUT_DIR / "tags_1x1.pdf"
        pages[0].save(
            pdf_path,
            save_all=True,
            append_images=pages[1:],
            resolution=DPI,
        )
        print(f"\nPDF (1×1 tags)  → {pdf_path}")
        print("Print at 100% / actual size on A4 — do NOT use 'fit to page'.")

    print(f"\nCOLMAP metadata → {CSV_PATH}")
    print(f"All output      → {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
