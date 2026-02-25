#!/usr/bin/env python3
"""
Register stationary cameras into a COLMAP reconstruction using manual
2D-3D correspondences.

Usage:
    python register_manual.py \
        --model_path /path/to/sparse/0 \
        --correspondences correspondences.json \
        --output_path /path/to/output \
        --stationary_dir ../stationary
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from collections import namedtuple
import cv2


# ============================================================================
# COLMAP binary model I/O
# ============================================================================

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageEntry = namedtuple("ImageEntry", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 4, 7: 5, 8: 8, 9: 5, 10: 12, 11: 12, 12: 4, 13: 5, 14: 3, 15: 4
}

CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "FULL_OPENCV", 6: "SIMPLE_RADIAL_FISHEYE",
    7: "RADIAL_FISHEYE", 8: "OPENCV_FISHEYE"
}


def read_next_bytes(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id = read_next_bytes(fid, 4, "i")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            width = read_next_bytes(fid, 8, "Q")[0]
            height = read_next_bytes(fid, 8, "Q")[0]
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = Camera(cam_id, model_id, width, height, params)
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = read_next_bytes(fid, 32, "dddd")
            tvec = read_next_bytes(fid, 24, "ddd")
            camera_id = read_next_bytes(fid, 4, "i")[0]
            name = b""
            while True:
                ch = fid.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2D):
                x, y = read_next_bytes(fid, 16, "dd")
                p3d_id = read_next_bytes(fid, 8, "q")[0]
                xys.append([x, y])
                point3D_ids.append(p3d_id)
            images[image_id] = ImageEntry(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            p3d_id = read_next_bytes(fid, 8, "Q")[0]
            xyz = read_next_bytes(fid, 24, "ddd")
            rgb = read_next_bytes(fid, 3, "BBB")
            error = read_next_bytes(fid, 8, "d")[0]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            image_ids = []
            point2D_idxs = []
            for _ in range(track_length):
                im_id, p2d_idx = read_next_bytes(fid, 8, "ii")
                image_ids.append(im_id)
                point2D_idxs.append(p2d_idx)
            points3D[p3d_id] = Point3D(p3d_id, xyz, rgb, error, image_ids, point2D_idxs)
    return points3D


def write_cameras_text(cameras, path):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras.keys()):
            cam = cameras[cam_id]
            model_name = CAMERA_MODEL_NAMES.get(cam.model, str(cam.model))
            params_str = " ".join(f"{p:.12g}" for p in cam.params)
            f.write(f"{cam.id} {model_name} {cam.width} {cam.height} {params_str}\n")


def write_images_text(images, path):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image_id in sorted(images.keys()):
            img = images[image_id]
            qw, qx, qy, qz = img.qvec
            tx, ty, tz = img.tvec
            f.write(f"{img.id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
                    f"{tx:.12g} {ty:.12g} {tz:.12g} {img.camera_id} {img.name}\n")
            pts_parts = []
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                pts_parts.append(f"{xy[0]:.6g} {xy[1]:.6g} {p3d_id}")
            f.write(" ".join(pts_parts) + "\n")


def write_points3D_text(points3D, path):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3D)}\n")
        for p3d_id in sorted(points3D.keys()):
            pt = points3D[p3d_id]
            track_parts = []
            for im_id, p2d_idx in zip(pt.image_ids, pt.point2D_idxs):
                track_parts.append(f"{im_id} {p2d_idx}")
            f.write(f"{pt.id} {pt.xyz[0]:.12g} {pt.xyz[1]:.12g} {pt.xyz[2]:.12g} "
                    f"{pt.rgb[0]} {pt.rgb[1]} {pt.rgb[2]} {pt.error:.12g} "
                    f"{' '.join(track_parts)}\n")


# ============================================================================
# Quaternion / camera utilities
# ============================================================================

def rotmat_to_qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    Rxx, Ryx, Rzx = R[0, 0], R[1, 0], R[2, 0]
    Rxy, Ryy, Rzy = R[0, 1], R[1, 1], R[2, 1]
    Rxz, Ryz, Rzz = R[0, 2], R[1, 2], R[2, 2]
    k = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def solve_pnp(pts_3d, pts_2d, K, dist_coeffs):
    """Try multiple PnP solvers on all correspondences (no RANSAC).

    Manual correspondences are assumed correct, so we use all points directly.
    """
    solvers = [
        ("SQPNP", cv2.SOLVEPNP_SQPNP),
        ("EPNP", cv2.SOLVEPNP_EPNP),
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
    ]
    for name, flag in solvers:
        try:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, K, dist_coeffs, flags=flag
            )
            if success:
                return success, rvec, tvec
        except cv2.error as e:
            print(f"    {name} failed: {e}")
            continue
    return False, None, None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Register stationary cameras using manual 2D-3D correspondences')
    parser.add_argument('--model_path', required=True, help='Path to COLMAP reconstruction (with cameras.bin, images.bin, points3D.bin)')
    parser.add_argument('--correspondences', required=True, help='JSON file with manual 2D-3D correspondences')
    parser.add_argument('--output_path', required=True, help='Output path for updated reconstruction (text format)')
    parser.add_argument('--stationary_dir', required=True, help='Directory with stationary camera images')
    parser.add_argument('--max_reproj_error', type=float, default=50.0, help='Maximum mean reprojection error (px) to accept a pose')
    args = parser.parse_args()

    # Load COLMAP reconstruction
    print(f"Loading reconstruction from {args.model_path}")
    cameras = read_cameras_binary(str(Path(args.model_path) / "cameras.bin"))
    images = read_images_binary(str(Path(args.model_path) / "images.bin"))
    points3D = read_points3D_binary(str(Path(args.model_path) / "points3D.bin"))
    print(f"  {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Load manual correspondences
    with open(args.correspondences) as f:
        corr_data = json.load(f)
    print(f"Loaded correspondences for {len(corr_data)} images")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    registered_count = 0
    next_image_id = max(images.keys()) + 1 if images else 1

    for stat_name, entry in corr_data.items():
        print(f"\n{'='*60}")
        print(f"Processing: {stat_name}")
        print(f"{'='*60}")

        pts_2d_raw = entry["points_2d"]
        pts_3d_raw = entry["points_3d"]

        # Skip incomplete entries
        has_todo = any(
            isinstance(p[0], str) or isinstance(p[1], str) or isinstance(p[2], str)
            for p in pts_3d_raw
        )
        if has_todo:
            print(f"  SKIPPING: 3D coordinates still have TODO placeholders")
            continue

        pts_2d = np.array(pts_2d_raw, dtype=np.float64)
        pts_3d = np.array(pts_3d_raw, dtype=np.float64)
        print(f"  {len(pts_2d)} manual correspondences")

        if len(pts_2d) < 4:
            print(f"  SKIPPING: Need at least 4 correspondences for PnP")
            continue

        # Get image dimensions
        stat_dir = Path(args.stationary_dir)
        stat_img_path = stat_dir / stat_name
        if not stat_img_path.exists():
            # Try searching
            matches = list(stat_dir.glob(f"*{stat_name}*"))
            if matches:
                stat_img_path = matches[0]
            else:
                print(f"  WARNING: Image not found at {stat_img_path}")
                continue

        img = cv2.imread(str(stat_img_path))
        H, W = img.shape[:2]
        print(f"  Image size: {W}x{H}")

        # Try many focal lengths at fine granularity.
        # solvePnP is cheap so we can afford a dense sweep.
        focal_candidates = [
            max(W, H) * f for f in np.arange(0.3, 3.05, 0.05)
        ]
        dist_coeffs = np.zeros(4)

        print(f"  Trying {len(focal_candidates)} focal lengths...")
        best_f = None
        best_error = float("inf")
        best_result = None

        for f_test in focal_candidates:
            K_test = np.array([[f_test, 0, W/2], [0, f_test, H/2], [0, 0, 1]], dtype=np.float64)
            success, rvec, tvec = solve_pnp(pts_3d, pts_2d, K_test, dist_coeffs)
            if not success:
                print(f"    f={f_test:.1f}: solver failed")
                continue
            projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K_test, dist_coeffs)
            err = np.mean(np.linalg.norm(projected.reshape(-1, 2) - pts_2d, axis=1))
            print(f"    f={f_test:.1f}: mean reproj error={err:.2f}px")
            if err < best_error:
                best_error = err
                best_f = f_test
                best_result = (rvec, tvec)

        if best_f is None:
            print(f"  FAILED: All solvers failed")
            continue
        if best_error > args.max_reproj_error:
            print(f"  FAILED: Best reprojection error {best_error:.2f}px exceeds threshold {args.max_reproj_error}px")
            continue

        K = np.array([[best_f, 0, W/2], [0, best_f, H/2], [0, 0, 1]], dtype=np.float64)
        rvec, tvec = best_result
        print(f"  Best focal length: {best_f:.1f} (mean reproj error={best_error:.2f}px)")

        # Refine using all manual correspondences
        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            pts_3d, pts_2d, K, dist_coeffs, rvec, tvec
        )

        # Convert to COLMAP format
        R, _ = cv2.Rodrigues(rvec_refined)
        qvec = rotmat_to_qvec(R)
        tvec_colmap = tvec_refined.flatten()
        print(f"  Pose: qvec={qvec}, tvec={tvec_colmap}")

        # Reprojection error after refinement (all correspondences)
        projected, _ = cv2.projectPoints(pts_3d, rvec_refined, tvec_refined, K, dist_coeffs)
        reproj_error = np.mean(np.linalg.norm(projected.reshape(-1, 2) - pts_2d, axis=1))
        print(f"  Mean reprojection error (refined, all pts): {reproj_error:.2f}px")

        # Add camera (SIMPLE_PINHOLE)
        new_cam_id = max(cameras.keys()) + 1 if cameras else 1
        cameras[new_cam_id] = Camera(new_cam_id, 0, W, H, (best_f, W/2, H/2))

        # Add image with all manual correspondences as observations (no point3D links)
        xys = [[pts_2d[i][0], pts_2d[i][1]] for i in range(len(pts_2d))]
        p3d_ids_for_img = [-1] * len(pts_2d)  # No 3D point links

        images[next_image_id] = ImageEntry(
            next_image_id, tuple(qvec), tuple(tvec_colmap),
            new_cam_id, stat_name, xys, p3d_ids_for_img
        )
        print(f"  Added as image {next_image_id}")
        next_image_id += 1
        registered_count += 1

    # Write output
    print(f"\n{'='*60}")
    print(f"Writing output to {args.output_path} (text format)")
    write_cameras_text(cameras, str(output_path / "cameras.txt"))
    write_images_text(images, str(output_path / "images.txt"))
    write_points3D_text(points3D, str(output_path / "points3D.txt"))

    print(f"\nRegistered {registered_count}/{len(corr_data)} stationary cameras")
    print(f"Total images in reconstruction: {len(images)}")


if __name__ == '__main__':
    main()
