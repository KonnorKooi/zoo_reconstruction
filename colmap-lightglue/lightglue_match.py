#!/usr/bin/env python3
"""
LightGlue matching for stationary cameras to handheld images.

This script:
1. Loads existing COLMAP database with SIFT features
2. Uses SuperPoint + LightGlue to find matches between stationary and handheld images
3. Injects SuperPoint keypoints for stationary images
4. Maps LightGlue matches to nearest SIFT keypoints in handheld images
5. Adds matches to database for COLMAP mapper

Usage:
    pip install lightglue torch torchvision

    python lightglue_match.py \
        --database_path /path/to/database.db \
        --stationary_dir /path/to/stationary \
        --handheld_dir /path/to/handheld
"""

import argparse
import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# Check for LightGlue
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, numpy_image_to_torch
except ImportError:
    print("ERROR: LightGlue not installed. Run:")
    print("  pip install lightglue")
    exit(1)


def image_ids_to_pair_id(image_id1, image_id2):
    """COLMAP's pair_id encoding."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2


def get_image_info(conn):
    """Get image_id -> name mapping from database."""
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, name FROM images")
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_keypoints(conn, image_id):
    """Get keypoints for an image from database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
        (image_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    rows, cols, data = row
    return np.frombuffer(data, dtype=np.float32).reshape(rows, cols)


def update_keypoints(conn, image_id, keypoints):
    """Update keypoints for an image (replace existing)."""
    cursor = conn.cursor()
    rows, cols = keypoints.shape
    cursor.execute(
        "INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
        (image_id, rows, cols, keypoints.astype(np.float32).tobytes()))


def update_descriptors(conn, image_id, descriptors):
    """Update descriptors for an image (placeholder for COLMAP schema)."""
    cursor = conn.cursor()
    rows, cols = descriptors.shape
    # Store as uint8 placeholder
    desc_uint8 = (descriptors * 127 + 128).clip(0, 255).astype(np.uint8)
    cursor.execute(
        "INSERT OR REPLACE INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
        (image_id, rows, cols, desc_uint8.tobytes()))


def add_matches(conn, image_id1, image_id2, matches):
    """Add matches between two images."""
    if len(matches) == 0:
        return
    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
        (pair_id, len(matches), 2, matches.astype(np.uint32).tobytes()))


def add_two_view_geometry(conn, image_id1, image_id2, matches):
    """Add two-view geometry for mapper."""
    if len(matches) == 0:
        return
    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    if image_id1 > image_id2:
        matches = matches[:, ::-1]
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO two_view_geometries "
        "(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pair_id, len(matches), 2, matches.astype(np.uint32).tobytes(),
         0, b'', b'', b'', b'', b''))


def find_nearest_keypoints(query_pts, keypoints, max_dist=5.0):
    """
    For each query point, find the nearest keypoint index.
    Returns indices and distances. Returns -1 for points with no nearby keypoint.
    """
    if len(query_pts) == 0 or len(keypoints) == 0:
        return np.array([]), np.array([])

    # Use only x, y coordinates
    kp_xy = keypoints[:, :2]

    indices = []
    distances = []
    for pt in query_pts:
        dists = np.linalg.norm(kp_xy - pt[:2], axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        if min_dist <= max_dist:
            indices.append(min_idx)
            distances.append(min_dist)
        else:
            indices.append(-1)
            distances.append(min_dist)

    return np.array(indices), np.array(distances)


def load_image_for_lightglue(path, device):
    """Load image for LightGlue."""
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor, img.size  # (W, H)


def extract_multiscale_features(extractor, img_path, device, max_keypoints,
                                 detection_threshold=0.001, scales=[1.0, 0.5, 2.0]):
    """
    Extract SuperPoint features at multiple scales and merge them.
    This helps get more features from stationary/far-away cameras.
    """
    from PIL import Image

    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    all_keypoints = []
    all_descriptors = []

    for scale in scales:
        # Resize image
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        if scale != 1.0:
            img_scaled = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            img_scaled = img

        # Convert to tensor
        img_np = np.array(img_scaled).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            feats = extractor.extract(img_tensor)

        kp = feats['keypoints'][0].cpu().numpy()  # (N, 2)
        desc = feats['descriptors'][0].cpu().numpy()  # (N, 256)

        # Scale keypoints back to original image coordinates
        if scale != 1.0:
            kp = kp / scale

        all_keypoints.append(kp)
        all_descriptors.append(desc)

    # Merge keypoints from all scales
    merged_kp = np.vstack(all_keypoints)
    merged_desc = np.vstack(all_descriptors)

    # Remove duplicate keypoints (within 2 pixel radius)
    if len(merged_kp) > 0:
        unique_mask = np.ones(len(merged_kp), dtype=bool)
        for i in range(len(merged_kp)):
            if not unique_mask[i]:
                continue
            dists = np.linalg.norm(merged_kp[i+1:] - merged_kp[i], axis=1)
            duplicates = np.where(dists < 2.0)[0] + i + 1
            unique_mask[duplicates] = False

        merged_kp = merged_kp[unique_mask]
        merged_desc = merged_desc[unique_mask]

    # Limit to max_keypoints (keep highest response if scores available)
    if len(merged_kp) > max_keypoints:
        indices = np.random.choice(len(merged_kp), max_keypoints, replace=False)
        merged_kp = merged_kp[indices]
        merged_desc = merged_desc[indices]

    return merged_kp, merged_desc, (orig_w, orig_h)


def main():
    parser = argparse.ArgumentParser(
        description='LightGlue matching for stationary to handheld images')
    parser.add_argument('--database_path', required=True,
                        help='Path to COLMAP database')
    parser.add_argument('--stationary_dir', required=True,
                        help='Directory with stationary camera images')
    parser.add_argument('--handheld_dir', required=True,
                        help='Directory with handheld camera images')
    parser.add_argument('--max_keypoint_dist', type=float, default=5.0,
                        help='Max distance to match LightGlue point to SIFT keypoint (pixels)')
    parser.add_argument('--min_matches', type=int, default=20,
                        help='Minimum matches to keep a pair')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_keypoints', type=int, default=8192,
                        help='Maximum SuperPoint keypoints to extract per image')
    parser.add_argument('--stationary_max_keypoints', type=int, default=32768,
                        help='Maximum keypoints for stationary images (higher for more features)')
    parser.add_argument('--stationary_detection_threshold', type=float, default=0.001,
                        help='SuperPoint detection threshold for stationary (lower = more features)')
    parser.add_argument('--multiscale', action='store_true',
                        help='Use multi-scale extraction for stationary images')
    parser.add_argument('--scales', type=str, default='0.5,1.0,2.0',
                        help='Comma-separated scales for multi-scale extraction')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse scales
    scales = [float(s) for s in args.scales.split(',')]

    # Load models
    print(f"Loading SuperPoint and LightGlue...")
    print(f"  Handheld: max_keypoints={args.max_keypoints}")
    print(f"  Stationary: max_keypoints={args.stationary_max_keypoints}, "
          f"detection_threshold={args.stationary_detection_threshold}")
    if args.multiscale:
        print(f"  Multi-scale enabled with scales: {scales}")

    # Extractor for handheld images (standard settings)
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)

    # Extractor for stationary images (more sensitive to get more features)
    extractor_stationary = SuperPoint(
        max_num_keypoints=args.stationary_max_keypoints,
        detection_threshold=args.stationary_detection_threshold
    ).eval().to(device)

    matcher = LightGlue(features='superpoint').eval().to(device)

    # Find images
    stationary_dir = Path(args.stationary_dir)
    handheld_dir = Path(args.handheld_dir)

    stat_images = sorted([p for p in stationary_dir.glob('*')
                          if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    hand_images = sorted([p for p in handheld_dir.glob('*')
                          if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    print(f"Found {len(stat_images)} stationary, {len(hand_images)} handheld images")

    if len(stat_images) == 0 or len(hand_images) == 0:
        print("ERROR: No images found")
        return

    # Open database
    conn = sqlite3.connect(args.database_path)
    image_info = get_image_info(conn)

    # Create name -> id mapping
    name_to_id = {name: img_id for img_id, name in image_info.items()}

    print(f"Database has {len(image_info)} images")

    # Process each stationary image
    total_matches_added = 0
    pairs_with_matches = 0

    for stat_path in stat_images:
        stat_name = stat_path.name

        # Find in database (check subdirectory paths too)
        stat_id = None
        for name, img_id in name_to_id.items():
            if name.endswith(stat_name) or stat_name in name:
                stat_id = img_id
                stat_db_name = name
                break

        if stat_id is None:
            print(f"WARNING: {stat_name} not found in database, skipping")
            continue

        print(f"\nProcessing stationary: {stat_name} (id={stat_id})")

        # Extract SuperPoint features for stationary image
        if args.multiscale:
            # Multi-scale extraction for better coverage
            stat_keypoints, stat_descriptors, (W_stat, H_stat) = extract_multiscale_features(
                extractor_stationary, stat_path, device,
                max_keypoints=args.stationary_max_keypoints,
                detection_threshold=args.stationary_detection_threshold,
                scales=scales
            )
            print(f"  Multi-scale SuperPoint extracted {len(stat_keypoints)} keypoints")

            # Build feats_stat dict for LightGlue matching
            feats_stat = {
                'keypoints': torch.from_numpy(stat_keypoints).unsqueeze(0).to(device),
                'descriptors': torch.from_numpy(stat_descriptors).unsqueeze(0).to(device),
                'image_size': torch.tensor([[H_stat, W_stat]]).to(device)
            }
        else:
            # Single-scale extraction with more sensitive extractor
            with torch.no_grad():
                img_stat, (W_stat, H_stat) = load_image_for_lightglue(stat_path, device)
                feats_stat = extractor_stationary.extract(img_stat)

            stat_keypoints = feats_stat['keypoints'][0].cpu().numpy()  # (N, 2)
            stat_descriptors = feats_stat['descriptors'][0].cpu().numpy()  # (N, 256)
            print(f"  SuperPoint extracted {len(stat_keypoints)} keypoints")

        # Update stationary image keypoints in database
        # COLMAP expects (x, y) or (x, y, scale, orientation)
        stat_kp_colmap = np.zeros((len(stat_keypoints), 2), dtype=np.float32)
        stat_kp_colmap[:, :2] = stat_keypoints
        update_keypoints(conn, stat_id, stat_kp_colmap)
        update_descriptors(conn, stat_id, stat_descriptors)

        # Match against each handheld image
        for hand_path in hand_images:
            hand_name = hand_path.name

            # Find in database
            hand_id = None
            for name, img_id in name_to_id.items():
                if name.endswith(hand_name) or hand_name in name:
                    hand_id = img_id
                    break

            if hand_id is None:
                continue

            # Get existing SIFT keypoints for handheld image
            hand_sift_kp = get_keypoints(conn, hand_id)
            if hand_sift_kp is None or len(hand_sift_kp) == 0:
                continue

            # Extract SuperPoint for handheld (temporary, for matching)
            with torch.no_grad():
                img_hand, _ = load_image_for_lightglue(hand_path, device)
                feats_hand = extractor.extract(img_hand)

                # Match with LightGlue
                matches_result = matcher({
                    'image0': feats_stat,
                    'image1': feats_hand
                })

            matches_lg = matches_result['matches'][0].cpu().numpy()  # (M, 2)

            if len(matches_lg) == 0:
                continue

            # Get matched points in pixel coordinates
            hand_superpoint_kp = feats_hand['keypoints'][0].cpu().numpy()

            matched_stat_pts = stat_keypoints[matches_lg[:, 0]]  # SuperPoint coords in stationary
            matched_hand_pts = hand_superpoint_kp[matches_lg[:, 1]]  # SuperPoint coords in handheld

            # Find nearest SIFT keypoints in handheld image
            hand_sift_indices, dists = find_nearest_keypoints(
                matched_hand_pts, hand_sift_kp, args.max_keypoint_dist)

            # Filter valid matches (where we found a nearby SIFT keypoint)
            valid_mask = hand_sift_indices >= 0

            if valid_mask.sum() < args.min_matches:
                continue

            # Create final matches: (SuperPoint idx in stationary, SIFT idx in handheld)
            final_matches = np.stack([
                matches_lg[valid_mask, 0],  # SuperPoint keypoint index in stationary
                hand_sift_indices[valid_mask]  # SIFT keypoint index in handheld
            ], axis=1).astype(np.uint32)

            # Add to database
            add_matches(conn, stat_id, hand_id, final_matches)
            add_two_view_geometry(conn, stat_id, hand_id, final_matches)

            total_matches_added += len(final_matches)
            pairs_with_matches += 1
            print(f"  -> {hand_name}: {len(final_matches)} matches "
                  f"(from {len(matches_lg)} LightGlue matches)")

    conn.commit()
    conn.close()

    print(f"\n=== Done ===")
    print(f"Pairs with matches: {pairs_with_matches}")
    print(f"Total matches added: {total_matches_added}")
    print(f"\nNow run COLMAP mapper:")
    print(f"  colmap mapper \\")
    print(f"    --database_path {args.database_path} \\")
    print(f"    --image_path <your_image_dir> \\")
    print(f"    --output_path <output_dir>")


if __name__ == '__main__':
    main()
