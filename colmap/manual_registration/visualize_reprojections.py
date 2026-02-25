#!/usr/bin/env python3
"""
Visualize reprojected 3D points on stationary camera images to diagnose
whether PnP poses are correct.

Usage:
    uv run python visualize_reprojections.py \
        --registered output/registered \
        --correspondences correspondences.json \
        --stationary_dir ../stationary \
        --output_dir output/reprojections
"""

import argparse
import json
import struct
import numpy as np
import cv2
from pathlib import Path
from collections import namedtuple

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageEntry = namedtuple("ImageEntry", ["id", "qvec", "tvec", "camera_id", "name"])

CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
}


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,   2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,   1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,   2*qy*qz + 2*qx*qw,   1 - 2*qx**2 - 2*qy**2],
    ])
    return R


def read_registered_images(path):
    """Read images.txt from the registered output (two lines per image)."""
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = tuple(float(x) for x in parts[1:5])
            tvec = tuple(float(x) for x in parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]
            images[image_id] = ImageEntry(image_id, qvec, tvec, camera_id, name)
            i += 2  # skip the points2D line
        else:
            i += 1
    return images


def read_registered_cameras(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = tuple(float(x) for x in parts[4:])
                cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered", required=True, help="Path to registered output (text format)")
    parser.add_argument("--correspondences", required=True, help="correspondences.json")
    parser.add_argument("--stationary_dir", required=True, help="Directory with stationary images")
    parser.add_argument("--output_dir", required=True, help="Output directory for visualization images")
    args = parser.parse_args()

    registered_path = Path(args.registered)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = read_registered_images(registered_path / "images.txt")
    cameras = read_registered_cameras(registered_path / "cameras.txt")

    with open(args.correspondences) as f:
        corr_data = json.load(f)

    stat_dir = Path(args.stationary_dir)

    # Find stationary cameras by name
    stat_images = {img.name: img for img in images.values() if img.name in corr_data}

    for name, entry in corr_data.items():
        if name not in stat_images:
            print(f"  {name}: not found in registered model, skipping")
            continue

        img_entry = stat_images[name]
        cam = cameras[img_entry.camera_id]

        # Build K from SIMPLE_PINHOLE params (f, cx, cy)
        f, cx, cy = cam.params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

        # Get pose: R, t  (world-to-camera)
        R = qvec_to_rotmat(img_entry.qvec)
        t = np.array(img_entry.tvec).reshape(3, 1)

        pts_3d = np.array(entry["points_3d"], dtype=np.float64)
        pts_2d = np.array(entry["points_2d"], dtype=np.float64)

        # Project 3D points
        pts_cam = R @ pts_3d.T + t  # 3xN
        pts_proj = (K @ pts_cam).T  # Nx3
        pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]  # Nx2

        # Load image
        img_path = stat_dir / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  {name}: could not load image")
            continue

        # Draw correspondences
        for i in range(len(pts_2d)):
            px_orig = tuple(int(round(x)) for x in pts_2d[i])
            px_proj = tuple(int(round(x)) for x in pts_proj[i])

            # Original click: green circle
            cv2.circle(img, px_orig, 8, (0, 255, 0), 2)
            cv2.putText(img, str(i), (px_orig[0]+6, px_orig[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Reprojected: red circle
            cv2.circle(img, px_proj, 6, (0, 0, 255), 2)
            cv2.putText(img, str(i), (px_proj[0]+6, px_proj[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Line between them
            cv2.line(img, px_orig, px_proj, (255, 0, 255), 1)

            err = np.linalg.norm(pts_2d[i] - pts_proj[i])
            print(f"  {name} pt{i}: orig={px_orig} proj={px_proj} err={err:.1f}px")

        # Legend
        cv2.putText(img, "GREEN=clicked  RED=reprojected", (10, img.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = output_dir / f"reproj_{name}"
        cv2.imwrite(str(out_path), img)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
