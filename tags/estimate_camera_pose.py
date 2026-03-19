"""
estimate_camera_pose.py

Detect Tag36h11 AprilTags in zoo camera images and output camera poses in
COLMAP World-to-Camera (W2C) format.

Usage:
    uv run estimate_camera_pose.py --images <folder> [--output poses.txt] [--vis-dir <folder>]

Output format (COLMAP images.txt):
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "opencv-contrib-python",
#   "numpy",
#   "Pillow",
# ]
# ///

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from tag_dict import build_custom_dictionary

# ---------------------------------------------------------------------------
# CAMERA INTRINSICS — Canon EOS 5D Mark IV + EF 24-105mm f/4L IS II USM
#
# Pixel density computed from EXIF Focal Plane Resolution:
#   Focal Plane X Resolution: 5719.148936 px/inch  →  225.163 px/mm
#   Focal Plane Y Resolution: 5728.900256 px/inch  →  225.547 px/mm
#
# fx = focal_length_mm * px_per_mm_x  (scaled to working resolution)
# fy = focal_length_mm * px_per_mm_y  (scaled to working resolution)
#
# Intrinsics are computed per-image from EXIF focal length because the dataset
# contains images shot at 24mm (20 images) and 31mm (8Y0A8436.JPG) — a single
# shared K matrix would be wrong for the 31mm shot.
#
# DISTORTION: dist = zeros — no lens calibration has been performed.
# The Canon EF 24-105mm f/4L IS II USM has measurable barrel distortion at 24mm
# and Digital Lens Optimizer was OFF during capture.
# Run cv2.calibrateCamera() with a checkerboard at 24mm and 31mm for best results.
# ---------------------------------------------------------------------------

# Pixel density from EXIF (px/mm, native 6720×4480 sensor)
_PX_PER_MM_X = 5719.148936 / 25.4   # 225.163 px/mm
_PX_PER_MM_Y = 5728.900256 / 25.4   # 225.547 px/mm

# Native sensor resolution
_NATIVE_W = 6720
_NATIVE_H = 4480

# Distortion: zeros until a proper calibration is done
dist = np.zeros((5,), dtype=np.float64)


def get_focal_length_mm(img_path: Path) -> float | None:
    """Read focal length in mm from EXIF. Returns None if unavailable."""
    try:
        with Image.open(img_path) as img:
            exif_data = img._getexif()
        if exif_data is None:
            return None
        for tag_id, value in exif_data.items():
            if TAGS.get(tag_id) == "FocalLength":
                # PIL returns an IFDRational; fall back to tuple for older Pillow
                return float(value) if hasattr(value, "numerator") else float(value[0]) / float(value[1])
    except Exception:
        pass
    return None


def compute_K(focal_mm: float, img_w: int, img_h: int) -> np.ndarray:
    """
    Build the 3×3 camera matrix for the given focal length and image size.
    Scales the native sensor pixel density to the working resolution.
    """
    scale_x = img_w / _NATIVE_W
    scale_y = img_h / _NATIVE_H
    fx = focal_mm * _PX_PER_MM_X * scale_x
    fy = focal_mm * _PX_PER_MM_Y * scale_y
    cx = img_w / 2.0
    cy = img_h / 2.0
    return np.array([
        [fx,  0.0, cx],
        [0.0,  fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

def build_detector_params() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.errorCorrectionRate = 1.0
    return params


# ---------------------------------------------------------------------------
# AprilTag half-size 3-D object points (tag-local frame, metres)
#
# Corner order that pupil_apriltags returns (counter-clockwise from top-left
# when looking at the tag face):
#   0: top-left     (-s, -s, 0)
#   1: top-right    (+s, -s, 0)
#   2: bottom-right (+s, +s, 0)
#   3: bottom-left  (-s, +s, 0)
#
# The tag plane is Z=0 in tag-local coordinates.  solvePnP will return the
# transform that maps these points into camera space (OpenCV: X-right, Y-down,
# Z-forward).
# ---------------------------------------------------------------------------
def tag_object_points(tag_size_m: float) -> np.ndarray:
    """Return the 4 corner 3-D object points for a tag of the given size."""
    s = tag_size_m / 2.0
    return np.array([
        [-s, -s, 0.0],
        [ s, -s, 0.0],
        [ s,  s, 0.0],
        [-s,  s, 0.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV rotation vector to a Hamilton quaternion [qw, qx, qy, qz]."""
    R, _ = cv2.Rodrigues(rvec)
    # Shepperd's method
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def average_poses(rvecs: list[np.ndarray], tvecs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Average multiple camera poses (each estimated from one tag).

    Translations are averaged directly.  Rotations are averaged by converting
    to quaternions, summing, and re-normalising (valid for small angular
    spread; good enough for tags close together in the same frame).
    """
    if len(rvecs) == 1:
        return rvecs[0], tvecs[0]

    # Average translation
    t_avg = np.mean(np.stack(tvecs, axis=0), axis=0)

    # Average rotation via quaternion mean
    quats = np.stack([rvec_to_quat(r) for r in rvecs], axis=0)
    # Ensure all quats are on the same hemisphere (flip if dot with first < 0)
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    q_avg = quats.mean(axis=0)
    q_avg /= np.linalg.norm(q_avg)

    # Convert back to rvec for consistency
    qw, qx, qy, qz = q_avg
    R_avg = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    r_avg, _ = cv2.Rodrigues(R_avg)
    return r_avg, t_avg


def detect_in_tiles(
    gray: np.ndarray,
    aruco_detector: cv2.aruco.ArucoDetector,
    grid_rows: int = 2,
    grid_cols: int = 2,
    overlap: float = 0.2,
) -> tuple[list, list]:
    """
    Run the detector on overlapping tiles, returning corners and ids with
    coordinates mapped back to full-image space.
    Returns (corners_list, ids_list) in the same format as detectMarkers.
    """
    h, w = gray.shape
    tile_w = int(w / grid_cols * (1.0 + overlap))
    tile_h = int(h / grid_rows * (1.0 + overlap))
    tile_w = min(tile_w, w)
    tile_h = min(tile_h, h)

    x_starts = [int(c * (w - tile_w) / max(grid_cols - 1, 1)) for c in range(grid_cols)]
    y_starts = [int(r * (h - tile_h) / max(grid_rows - 1, 1)) for r in range(grid_rows)]

    all_corners: list = []
    all_ids: list = []
    for y0 in y_starts:
        for x0 in x_starts:
            tile = gray[y0:y0 + tile_h, x0:x0 + tile_w]
            t_corners, t_ids, _ = aruco_detector.detectMarkers(tile)
            if t_ids is None:
                continue
            for corners, tag_id in zip(t_corners, t_ids.flatten()):
                shifted = corners.copy()
                shifted[0, :, 0] += x0
                shifted[0, :, 1] += y0
                all_corners.append(shifted)
                all_ids.append(tag_id)
    return all_corners, all_ids


def merge_detections(
    full_corners: list, full_ids: list,
    tile_corners: list, tile_ids: list,
) -> tuple[list, list[int]]:
    """
    Merge full-image and tiled detections, keeping the largest detection
    (by quad area) when the same tag ID appears in multiple passes.
    """
    def quad_area(corners) -> float:
        return float(cv2.contourArea(corners[0].astype(np.float32)))

    # {tag_id: (corners, area)} — one entry per unique tag
    best: dict[int, tuple[object, float]] = {}
    for corners, tid in zip(full_corners + tile_corners, full_ids + tile_ids):
        area = quad_area(corners)
        if tid not in best or area > best[tid][1]:
            best[tid] = (corners, area)

    ids = list(best.keys())
    corners = [best[i][0] for i in ids]
    return corners, ids


def draw_tag_axes(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  tag_size_m: float, tag_id: int, K: np.ndarray) -> np.ndarray:
    """Draw coordinate axes and tag ID on the image for one detected tag."""
    axis_len = tag_size_m * 0.6

    # Project axes endpoints
    axes_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],   # X — red
        [0, axis_len, 0],   # Y — green
        [0, 0, -axis_len],  # Z — blue  (points out of tag face toward camera)
    ])
    pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    pts = pts.reshape(-1, 2).astype(int)

    origin = tuple(pts[0])
    cv2.arrowedLine(img, origin, tuple(pts[1]), (0, 0, 255), 2, tipLength=0.2)   # X red
    cv2.arrowedLine(img, origin, tuple(pts[2]), (0, 255, 0), 2, tipLength=0.2)   # Y green
    cv2.arrowedLine(img, origin, tuple(pts[3]), (255, 0, 0), 2, tipLength=0.2)   # Z blue

    cv2.putText(img, f"ID {tag_id}", (origin[0] + 5, origin[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_tag_configs(csv_path: Path) -> dict[int, float]:
    """Return {tag_id: size_in_metres} from tag_configs.csv."""
    configs: dict[int, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs[int(row["Tag_ID"])] = float(row["Actual_Size_In_Meters"])
    return configs


def process_image(
    img_path: Path,
    detector: cv2.aruco.ArucoDetector,
    tag_configs: dict[int, float],
    image_id: int,
    vis_dir: Path | None,
) -> str | None:
    """
    Detect tags in one image, estimate camera pose, and return a COLMAP
    images.txt line.  Returns None if no known tags are found.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [WARN] Could not read {img_path.name}", file=sys.stderr)
        return None

    img_h, img_w = img_bgr.shape[:2]

    # Build per-image K from EXIF focal length
    focal_mm = get_focal_length_mm(img_path)
    if focal_mm is None:
        print(f"  [WARN] Could not read focal length from EXIF for {img_path.name} — skipping", file=sys.stderr)
        return None
    K = compute_K(focal_mm, img_w, img_h)
    print(f"  [CAM] focal={focal_mm}mm  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Pass 1: full-image detection
    full_corners, full_ids, _ = detector.detectMarkers(gray)
    full_corners = list(full_corners)
    full_ids = full_ids.flatten().tolist() if full_ids is not None else []

    # Pass 2: tiled 2×2 detection — recovers small/distant background tags
    tile_corners, tile_ids = detect_in_tiles(gray, detector, grid_rows=2, grid_cols=2, overlap=0.2)

    all_corners, all_ids = merge_detections(full_corners, full_ids, tile_corners, tile_ids)
    print(f"  [DET] full={len(full_ids)}  tiles={len(tile_ids)}  merged={len(all_ids)}")

    if not all_ids:
        print(f"  [INFO] No tags detected in {img_path.name}")
        return None

    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    used_ids: list[int] = []

    for corners, tag_id in zip(all_corners, all_ids):
        if tag_id not in tag_configs:
            print(f"  [WARN] Detected tag {tag_id} not in tag_configs — skipping")
            continue

        tag_size = tag_configs[tag_id]
        obj_pts = tag_object_points(tag_size)
        img_pts = corners[0].astype(np.float64)  # shape (4, 2)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            print(f"  [WARN] solvePnP failed for tag {tag_id}")
            continue

        rvecs.append(rvec)
        tvecs.append(tvec)
        used_ids.append(tag_id)

        if vis_dir is not None:
            draw_tag_axes(img_bgr, rvec, tvec, tag_size, tag_id, K)

    if not rvecs:
        return None

    # --- Average pose across all detected tags ---
    rvec_avg, tvec_avg = average_poses(rvecs, tvecs)
    qw, qx, qy, qz = rvec_to_quat(rvec_avg)
    tx, ty, tz = tvec_avg.flatten()

    print(f"  [OK]  {img_path.name}  tags={used_ids}  "
          f"t=[{tx:.3f}, {ty:.3f}, {tz:.3f}]")

    # Save visualisation
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / img_path.name
        cv2.imwrite(str(out_path), img_bgr)

    # COLMAP images.txt line
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    camera_id = 1  # single shared camera model
    return (f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.9f} {ty:.9f} {tz:.9f} {camera_id} {img_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect AprilTags and estimate COLMAP camera poses."
    )
    parser.add_argument("--images", required=True, type=Path,
                        help="Folder containing input images (jpg/png).")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "apriltag_output" / "tag_configs.csv",
                        help="Path to tag_configs.csv (default: apriltag_output/tag_configs.csv).")
    parser.add_argument("--output", type=Path, default=Path("poses.txt"),
                        help="Output file for COLMAP-format poses (default: poses.txt).")
    parser.add_argument("--vis-dir", type=Path, default=None,
                        help="If set, save visualisation images to this folder.")
    parser.add_argument("--exts", nargs="+", default=["jpg", "jpeg", "png"],
                        help="Image file extensions to process.")
    args = parser.parse_args()

    # Load configs
    if not args.config.exists():
        sys.exit(f"[ERROR] tag_configs.csv not found at {args.config}")
    tag_configs = load_tag_configs(args.config)
    print(f"Loaded {len(tag_configs)} tag size entries from {args.config}")

    # Collect images
    image_paths: list[Path] = []
    for ext in args.exts:
        image_paths.extend(sorted(args.images.glob(f"*.{ext}")))
        image_paths.extend(sorted(args.images.glob(f"*.{ext.upper()}")))
    if not image_paths:
        sys.exit(f"[ERROR] No images found in {args.images}")
    print(f"Found {len(image_paths)} image(s) to process.")

    # Build custom ArUco dictionary matching the printed tags (IDs 0-34).
    # Uses the same spiral bit encoding as generate_apriltags.py / visualize_apriltags.py.
    print("Building custom tag dictionary...", end=" ", flush=True)
    dictionary = build_custom_dictionary()
    params = build_detector_params()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    print("done.")

    lines: list[str] = []
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {img_path.name}")
        line = process_image(img_path, detector, tag_configs, idx, args.vis_dir)
        if line is not None:
            lines.append(line)

    # Write output
    header = (
        "# COLMAP images.txt — camera poses estimated from AprilTags\n"
        "# Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "# Coordinate system: OpenCV / COLMAP W2C  (X-right, Y-down, Z-forward)\n"
        f"# {len(lines)} images with detected tags\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line + "\n")

    print(f"\nWrote {len(lines)} pose(s) to {args.output}")
    if args.vis_dir:
        print(f"Visualisations saved to {args.vis_dir}")


if __name__ == "__main__":
    main()
