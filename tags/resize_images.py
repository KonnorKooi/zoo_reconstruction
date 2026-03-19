"""
Resize Canon EOS 5D Mark IV RAW-converted JPEGs for AprilTag camera pipeline.

Camera: Canon EOS 5D Mark IV
Native resolution: 6720 x 4480 (30MP, 3:2 aspect ratio)
Target: ~1000px on the long edge, preserving aspect ratio, no warping.
Output: 1000 x 667 (landscape) or 667 x 1000 (portrait)
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow",
# ]
# ///

import argparse
import sys
from pathlib import Path
from PIL import Image

DEFAULT_INPUT_DIR  = Path(__file__).parent / "classroom_data" / "raw"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "classroom_data" / "resized"
DEFAULT_LONG_EDGE  = 1000  # px
DEFAULT_QUALITY    = 95    # JPEG quality 0-100


def analyze_aspect(img: Image.Image, path: Path) -> None:
    w, h = img.size
    ratio = w / h
    orientation = "landscape" if w >= h else "portrait"
    print(f"  original : {w} x {h}  |  ratio {ratio:.4f} ({orientation})")


def compute_target_size(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """Scale so the long edge == long_edge, preserving aspect ratio exactly."""
    if w >= h:
        new_w = long_edge
        new_h = round(h * long_edge / w)
    else:
        new_h = long_edge
        new_w = round(w * long_edge / h)
    return new_w, new_h


def resize_images(input_dir: Path, output_dir: Path, long_edge: int, quality: int) -> None:
    extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    images = [p for p in input_dir.iterdir() if p.suffix in extensions]

    if not images:
        print(f"No JPEG images found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Canon EOS 5D Mark IV image resizer")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Target long edge: {long_edge}px  |  JPEG quality: {quality}")
    print(f"Found {len(images)} image(s)\n")

    for img_path in sorted(images):
        print(f"[{img_path.name}]")
        with Image.open(img_path) as img:
            analyze_aspect(img, img_path)
            exif = img.info.get("exif", b"")  # preserve original EXIF
            new_w, new_h = compute_target_size(*img.size, long_edge)
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"  resized  : {new_w} x {new_h}")

        out_path = output_dir / img_path.name
        resized.save(out_path, "JPEG", quality=quality, exif=exif)
        in_kb  = img_path.stat().st_size / 1024
        out_kb = out_path.stat().st_size / 1024
        print(f"  size     : {in_kb:.0f} KB -> {out_kb:.0f} KB  ({out_kb/in_kb*100:.1f}%)")
        print()

    print(f"Done. {len(images)} image(s) saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize Canon EOS 5D Mark IV JPEGs for the AprilTag camera pipeline."
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=DEFAULT_INPUT_DIR,
        metavar="DIR", help=f"Input directory of JPEGs (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        metavar="DIR", help=f"Output directory for resized images (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-l", "--long-edge", type=int, default=DEFAULT_LONG_EDGE,
        metavar="PX", help=f"Target size of the long edge in pixels (default: {DEFAULT_LONG_EDGE})"
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=DEFAULT_QUALITY,
        metavar="0-100", help=f"JPEG output quality (default: {DEFAULT_QUALITY})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resize_images(args.input, args.output, args.long_edge, args.quality)
