#!/usr/bin/env python3
"""
Direct PnP registration of stationary cameras into a COLMAP reconstruction.

Instead of injecting matches into the database and hoping image_registrator works,
this script:
  1. Matches stationary images to handheld images via LightGlue (SuperPoint)
  2. Projects known 3D points onto handheld images using known camera poses
  3. Bridges SuperPoint matches to 3D points via nearest projected point
  4. Collects 2D (stationary pixel) <-> 3D (world point) correspondences
  5. Solves PnP + RANSAC to estimate the stationary camera pose directly
  6. Writes the updated reconstruction in text format
"""

import argparse
import struct
import sqlite3
import numpy as np
from pathlib import Path
from collections import namedtuple
from PIL import Image as PILImage
import torch
import cv2

try:
    from lightglue import LightGlue, SuperPoint
except ImportError:
    print("ERROR: LightGlue not installed.")
    exit(1)


# ============================================================================
# COLMAP binary model I/O
# ============================================================================

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageEntry = namedtuple("ImageEntry", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# Number of parameters per camera model
CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 4, 7: 5, 8: 8, 9: 5, 10: 12, 11: 12, 12: 4, 13: 5, 14: 3, 15: 4
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


CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "FULL_OPENCV", 6: "SIMPLE_RADIAL_FISHEYE",
    7: "RADIAL_FISHEYE", 8: "OPENCV_FISHEYE"
}


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
# Quaternion utilities (COLMAP convention: qw, qx, qy, qz)
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


def get_camera_matrix(camera):
    """Build intrinsic matrix K from COLMAP camera."""
    if camera.model == 0:  # SIMPLE_PINHOLE
        f, cx, cy = camera.params
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.zeros(4)
    elif camera.model == 1:  # PINHOLE
        fx, fy, cx, cy = camera.params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.zeros(4)
    elif camera.model == 2:  # SIMPLE_RADIAL
        f, cx, cy, k = camera.params
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k, 0, 0, 0])
    elif camera.model == 3:  # RADIAL
        f, cx, cy, k1, k2 = camera.params
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k1, k2, 0, 0])
    elif camera.model == 4:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([k1, k2, p1, p2])
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")


# ============================================================================
# Build 3D point lookup from reconstruction
# ============================================================================

def qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])


def project_3d_points_to_image(points3D, image, camera):
    """
    Project all visible 3D points onto a handheld image using known pose.

    Returns:
        projected_xy: np.array of shape (N, 2) - pixel coordinates
        point3D_ids: list of point3D IDs
        world_xyz: np.array of shape (N, 3) - world coordinates
    """
    R = qvec_to_rotmat(image.qvec)
    t = np.array(image.tvec)
    K, dist = get_camera_matrix(camera)

    # Collect 3D points visible in this image
    visible_xyz = []
    visible_ids = []
    for kp_idx, p3d_id in enumerate(image.point3D_ids):
        if p3d_id != -1 and p3d_id in points3D:
            visible_xyz.append(points3D[p3d_id].xyz)
            visible_ids.append(p3d_id)

    if len(visible_xyz) == 0:
        return np.zeros((0, 2)), [], np.zeros((0, 3))

    world_xyz = np.array(visible_xyz)

    # Project: x = K * (R * X + t)
    cam_pts = (R @ world_xyz.T).T + t  # (N, 3)
    # Filter points behind camera
    in_front = cam_pts[:, 2] > 0
    cam_pts = cam_pts[in_front]
    world_xyz = world_xyz[in_front]
    visible_ids = [pid for pid, f in zip(visible_ids, in_front) if f]

    if len(cam_pts) == 0:
        return np.zeros((0, 2)), [], np.zeros((0, 3))

    # Project to pixel coordinates (ignoring distortion for now)
    proj_x = K[0, 0] * cam_pts[:, 0] / cam_pts[:, 2] + K[0, 2]
    proj_y = K[1, 1] * cam_pts[:, 1] / cam_pts[:, 2] + K[1, 2]
    projected_xy = np.stack([proj_x, proj_y], axis=1)

    return projected_xy, visible_ids, world_xyz


# ============================================================================
# Feature extraction helpers
# ============================================================================

def load_image_tensor(path, device):
    img = PILImage.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, img.size  # (W, H)


def extract_multiscale(extractor, img_path, device, max_keypoints,
                       detection_threshold=0.0005, scales=[1.0]):
    img = PILImage.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    all_kp, all_desc = [], []
    for scale in scales:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_scaled = img.resize((new_w, new_h), PILImage.LANCZOS) if scale != 1.0 else img
        img_np = np.array(img_scaled).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = extractor.extract(img_tensor)
        kp = feats['keypoints'][0].cpu().numpy()
        desc = feats['descriptors'][0].cpu().numpy()
        if scale != 1.0:
            kp = kp / scale
        all_kp.append(kp)
        all_desc.append(desc)

    merged_kp = np.vstack(all_kp)
    merged_desc = np.vstack(all_desc)

    # Deduplicate within 2px
    if len(merged_kp) > 0:
        unique = np.ones(len(merged_kp), dtype=bool)
        for i in range(len(merged_kp)):
            if not unique[i]:
                continue
            dists = np.linalg.norm(merged_kp[i+1:] - merged_kp[i], axis=1)
            unique[np.where(dists < 2.0)[0] + i + 1] = False
        merged_kp = merged_kp[unique]
        merged_desc = merged_desc[unique]

    if len(merged_kp) > max_keypoints:
        idx = np.random.choice(len(merged_kp), max_keypoints, replace=False)
        merged_kp = merged_kp[idx]
        merged_desc = merged_desc[idx]

    return merged_kp, merged_desc, (orig_w, orig_h)


# ============================================================================
# Database helpers
# ============================================================================

def get_db_image_info(db_path):
    """Get image_id -> name mapping from COLMAP database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, name FROM images")
    info = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return info



# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Direct PnP registration of stationary cameras')
    parser.add_argument('--model_path', required=True, help='Path to COLMAP reconstruction (with cameras.bin, images.bin, points3D.bin)')
    parser.add_argument('--database_path', required=True, help='Path to COLMAP database')
    parser.add_argument('--output_path', required=True, help='Output path for updated reconstruction')
    parser.add_argument('--stationary_dir', required=True, help='Directory with stationary camera images')
    parser.add_argument('--handheld_dir', required=True, help='Directory with handheld camera images')
    parser.add_argument('--max_keypoint_dist', type=float, default=25.0, help='Max distance for SIFT-SuperPoint bridging')
    parser.add_argument('--min_correspondences', type=int, default=12, help='Minimum 2D-3D correspondences for PnP')
    parser.add_argument('--pnp_reproj_threshold', type=float, default=8.0, help='RANSAC reprojection threshold in pixels')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stationary_max_keypoints', type=int, default=32768)
    parser.add_argument('--stationary_detection_threshold', type=float, default=0.0005)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--scales', type=str, default='0.25,0.5,0.75,1.0,1.5,2.0,3.0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scales = [float(s) for s in args.scales.split(',')]

    # Load COLMAP reconstruction
    print(f"Loading reconstruction from {args.model_path}")
    cameras = read_cameras_binary(str(Path(args.model_path) / "cameras.bin"))
    images = read_images_binary(str(Path(args.model_path) / "images.bin"))
    points3D = read_points3D_binary(str(Path(args.model_path) / "points3D.bin"))
    print(f"  {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")

    # Count total 3D-linked observations
    total_obs = sum(1 for img in images.values() for p3d_id in img.point3D_ids if p3d_id != -1)
    print(f"  {total_obs} observations with known 3D points across all images")

    # Load database image info (maps image names to IDs)
    db_info = get_db_image_info(args.database_path)
    db_name_to_id = {name: img_id for img_id, name in db_info.items()}

    # Load LightGlue models
    print("Loading SuperPoint and LightGlue...")
    extractor = SuperPoint(max_num_keypoints=8192).eval().to(device)
    extractor_stationary = SuperPoint(
        max_num_keypoints=args.stationary_max_keypoints,
        detection_threshold=args.stationary_detection_threshold
    ).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    # Find images
    stationary_dir = Path(args.stationary_dir)
    handheld_dir = Path(args.handheld_dir)
    stat_images = sorted([p for p in stationary_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    hand_images = sorted([p for p in handheld_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    print(f"Found {len(stat_images)} stationary, {len(hand_images)} handheld images")

    # Map handheld filenames to their image IDs in the reconstruction
    recon_name_to_id = {img.name: img.id for img in images.values()}

    # We need to figure out which camera model to use for stationary images
    # Look up from the database
    stat_db_info = {}
    for sp in stat_images:
        for name, img_id in db_name_to_id.items():
            if name.endswith(sp.name) or sp.name in name:
                stat_db_info[sp.name] = img_id
                break

    # Get camera_id for stationary images from database
    conn = sqlite3.connect(args.database_path)
    cursor = conn.cursor()
    stat_camera_ids = {}
    for stat_name, stat_db_id in stat_db_info.items():
        cursor.execute("SELECT camera_id FROM images WHERE image_id = ?", (stat_db_id,))
        row = cursor.fetchone()
        if row:
            stat_camera_ids[stat_name] = row[0]
    conn.close()

    # Copy existing model files to output, we'll modify images.bin
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each stationary image
    registered_count = 0
    next_image_id = max(images.keys()) + 1 if images else 1

    for stat_path in stat_images:
        stat_name = stat_path.name
        print(f"\n{'='*60}")
        print(f"Processing stationary: {stat_name}")
        print(f"{'='*60}")

        # Extract SuperPoint features for stationary image
        if args.multiscale:
            stat_kp, stat_desc, (W_stat, H_stat) = extract_multiscale(
                extractor_stationary, stat_path, device,
                max_keypoints=args.stationary_max_keypoints,
                detection_threshold=args.stationary_detection_threshold,
                scales=scales
            )
            print(f"  Multi-scale SuperPoint: {len(stat_kp)} keypoints")
        else:
            with torch.no_grad():
                img_t, (W_stat, H_stat) = load_image_tensor(stat_path, device)
                feats = extractor_stationary.extract(img_t)
            stat_kp = feats['keypoints'][0].cpu().numpy()
            stat_desc = feats['descriptors'][0].cpu().numpy()
            print(f"  SuperPoint: {len(stat_kp)} keypoints")

        stat_feats = {
            'keypoints': torch.from_numpy(stat_kp).unsqueeze(0).to(device),
            'descriptors': torch.from_numpy(stat_desc).unsqueeze(0).to(device),
            'image_size': torch.tensor([[H_stat, W_stat]]).to(device),
        }

        # Collect 2D-3D correspondences across all handheld images
        all_pts_2d = []  # pixel coords in stationary image
        all_pts_3d = []  # world 3D coords
        all_p3d_ids = []  # point3D IDs for bookkeeping
        match_details = []

        for hand_path in hand_images:
            hand_name = hand_path.name

            # Find this handheld image in the reconstruction
            hand_recon_id = recon_name_to_id.get(hand_name)
            if hand_recon_id is None:
                continue

            hand_img = images[hand_recon_id]
            hand_cam = cameras[hand_img.camera_id]

            # Project all visible 3D points onto this handheld image
            proj_xy, proj_p3d_ids, proj_xyz = project_3d_points_to_image(
                points3D, hand_img, hand_cam)

            if len(proj_xy) == 0:
                continue

            # Extract SuperPoint for handheld and match via LightGlue
            with torch.no_grad():
                img_hand, _ = load_image_tensor(hand_path, device)
                feats_hand = extractor.extract(img_hand)
                matches_res = matcher({'image0': stat_feats, 'image1': feats_hand})
                matches_lg = matches_res['matches'][0].cpu().numpy()

            if len(matches_lg) == 0:
                continue

            hand_sp_kp = feats_hand['keypoints'][0].cpu().numpy()

            # For each LightGlue match, find nearest PROJECTED 3D point
            # (no SIFT bridging — uses geometrically exact projections)
            pair_2d = []
            pair_3d = []
            pair_ids = []

            for match in matches_lg:
                stat_idx, hand_sp_idx = match[0], match[1]
                hand_sp_pt = hand_sp_kp[hand_sp_idx]

                # Find nearest projected 3D point to this SuperPoint keypoint
                dists = np.linalg.norm(proj_xy - hand_sp_pt[:2], axis=1)
                min_idx = np.argmin(dists)

                if dists[min_idx] <= args.max_keypoint_dist:
                    pair_2d.append(stat_kp[stat_idx])
                    pair_3d.append(proj_xyz[min_idx])
                    pair_ids.append(proj_p3d_ids[min_idx])

            if len(pair_2d) > 0:
                all_pts_2d.extend(pair_2d)
                all_pts_3d.extend(pair_3d)
                all_p3d_ids.extend(pair_ids)
                match_details.append((hand_name, len(pair_2d), len(matches_lg)))

        # Report matching
        for hand_name, n_3d, n_lg in match_details:
            print(f"  -> {hand_name}: {n_3d} 2D-3D correspondences (from {n_lg} LightGlue matches)")

        pts_2d = np.array(all_pts_2d, dtype=np.float64)
        pts_3d = np.array(all_pts_3d, dtype=np.float64)
        p3d_ids = all_p3d_ids

        # Deduplicate: if the same 3D point appears multiple times, keep the one
        # with the smallest index (first occurrence)
        if len(p3d_ids) > 0:
            seen = {}
            unique_mask = []
            for i, pid in enumerate(p3d_ids):
                if pid not in seen:
                    seen[pid] = i
                    unique_mask.append(True)
                else:
                    unique_mask.append(False)
            unique_mask = np.array(unique_mask)
            pts_2d = pts_2d[unique_mask]
            pts_3d = pts_3d[unique_mask]
            p3d_ids = [pid for pid, keep in zip(p3d_ids, unique_mask) if keep]

        print(f"\n  Total unique 2D-3D correspondences: {len(pts_2d)}")

        if len(pts_2d) < args.min_correspondences:
            print(f"  SKIPPING: Not enough correspondences (need {args.min_correspondences})")
            continue

        # Estimate camera intrinsics for PnP
        # Try multiple focal lengths and pick the one with the most inliers
        dist_coeffs = np.zeros(4)
        known_camera = False

        if stat_name in stat_camera_ids:
            cam_id = stat_camera_ids[stat_name]
            if cam_id in cameras:
                K, dist_coeffs = get_camera_matrix(cameras[cam_id])
                known_camera = True
                print(f"  Using existing camera {cam_id} from reconstruction")

        if not known_camera:
            # No EXIF data, try a range of focal lengths
            focal_candidates = [
                max(W_stat, H_stat) * f for f in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
            ]
            print(f"  No EXIF data. Trying {len(focal_candidates)} focal lengths...")

            best_f = None
            best_inlier_count = 0
            best_result = None

            for f_test in focal_candidates:
                K_test = np.array([[f_test, 0, W_stat/2], [0, f_test, H_stat/2], [0, 0, 1]], dtype=np.float64)
                success_test, rvec_test, tvec_test, inliers_test = cv2.solvePnPRansac(
                    pts_3d, pts_2d, K_test, dist_coeffs,
                    iterationsCount=10000,
                    reprojectionError=args.pnp_reproj_threshold,
                    flags=cv2.SOLVEPNP_SQPNP
                )
                n_inliers = len(inliers_test) if success_test and inliers_test is not None else 0
                print(f"    f={f_test:.1f}: {n_inliers} inliers")
                if n_inliers > best_inlier_count:
                    best_inlier_count = n_inliers
                    best_f = f_test
                    best_result = (success_test, rvec_test, tvec_test, inliers_test)

            if best_f is None or best_inlier_count < args.min_correspondences:
                print(f"  FAILED: No focal length produced enough inliers")
                continue

            K = np.array([[best_f, 0, W_stat/2], [0, best_f, H_stat/2], [0, 0, 1]], dtype=np.float64)
            print(f"  Best focal length: {best_f:.1f} ({best_inlier_count} inliers)")
            success, rvec, tvec, inliers = best_result
        else:
            # Solve PnP + RANSAC with known intrinsics
            print(f"  Running PnP RANSAC (threshold={args.pnp_reproj_threshold}px)...")
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, K, dist_coeffs,
                iterationsCount=10000,
                reprojectionError=args.pnp_reproj_threshold,
                flags=cv2.SOLVEPNP_SQPNP
            )

        if not success or inliers is None:
            print(f"  FAILED: PnP did not converge")
            continue

        num_inliers = len(inliers)
        print(f"  PnP succeeded: {num_inliers}/{len(pts_2d)} inliers")

        # Refine with inliers only
        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            pts_3d[inliers.flatten()],
            pts_2d[inliers.flatten()],
            K, dist_coeffs, rvec, tvec
        )

        # Convert to COLMAP format
        R, _ = cv2.Rodrigues(rvec_refined)
        qvec = rotmat_to_qvec(R)
        tvec_colmap = tvec_refined.flatten()

        print(f"  Pose: qvec={qvec}, tvec={tvec_colmap}")

        # Compute reprojection error on inliers
        inlier_pts_3d = pts_3d[inliers.flatten()]
        inlier_pts_2d = pts_2d[inliers.flatten()]
        projected, _ = cv2.projectPoints(inlier_pts_3d, rvec_refined, tvec_refined, K, dist_coeffs)
        reproj_error = np.mean(np.linalg.norm(projected.reshape(-1, 2) - inlier_pts_2d, axis=1))
        print(f"  Mean reprojection error: {reproj_error:.2f}px")

        # Add camera to reconstruction if needed
        if stat_name in stat_camera_ids:
            cam_id = stat_camera_ids[stat_name]
            if cam_id not in cameras:
                # Add a new camera based on estimated intrinsics
                new_cam_id = max(cameras.keys()) + 1 if cameras else 1
                cameras[new_cam_id] = Camera(new_cam_id, 0, W_stat, H_stat,
                                             (K[0,0], K[0,2], K[1,2]))
                cam_id = new_cam_id
        else:
            new_cam_id = max(cameras.keys()) + 1 if cameras else 1
            cameras[new_cam_id] = Camera(new_cam_id, 0, W_stat, H_stat,
                                         (K[0,0], K[0,2], K[1,2]))
            cam_id = new_cam_id

        # Build 2D observations for this image (inlier correspondences)
        inlier_mask = np.zeros(len(pts_2d), dtype=bool)
        inlier_mask[inliers.flatten()] = True
        xys = []
        point3D_ids_for_img = []
        for i in range(len(pts_2d)):
            if inlier_mask[i]:
                xys.append([pts_2d[i][0], pts_2d[i][1]])
                point3D_ids_for_img.append(p3d_ids[i])

        # Add to reconstruction
        images[next_image_id] = ImageEntry(
            next_image_id, tuple(qvec), tuple(tvec_colmap),
            cam_id, stat_name, xys, point3D_ids_for_img
        )
        print(f"  Added as image {next_image_id} with {len(xys)} observations")
        next_image_id += 1
        registered_count += 1

    # Write output in text format (avoids COLMAP 3.14 frames/rigs binary issues)
    print(f"\n{'='*60}")
    print(f"Writing output to {args.output_path} (text format)")
    write_cameras_text(cameras, str(output_path / "cameras.txt"))
    write_images_text(images, str(output_path / "images.txt"))
    write_points3D_text(points3D, str(output_path / "points3D.txt"))

    print(f"\n{'='*60}")
    print(f"Done! Registered {registered_count}/{len(stat_images)} stationary cameras")
    print(f"Total images in reconstruction: {len(images)}")
    print(f"Output: {args.output_path}")


if __name__ == '__main__':
    main()
