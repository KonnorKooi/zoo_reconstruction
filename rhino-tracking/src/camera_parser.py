"""
COLMAP camera parser for reading camera intrinsics and extrinsics.
"""

from typing import Dict, Tuple
import numpy as np
from pathlib import Path

from .data_structures import CameraIntrinsics, CameraPose


def load_cameras(filepath: str) -> Dict[int, CameraIntrinsics]:
    """Load camera intrinsics from COLMAP cameras.txt file.

    Expected format:
        # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
        1 PINHOLE 1009 669 672.29 672.29 504.5 334.5

    Args:
        filepath: Path to cameras.txt file

    Returns:
        Dictionary mapping camera_id to CameraIntrinsics

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Camera file not found: {filepath}")

    cameras = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])

                # Only support PINHOLE model for now
                if model != 'PINHOLE':
                    print(f"Warning: Skipping camera {camera_id} with model {model} (only PINHOLE supported)")
                    continue

                # PINHOLE params: fx, fy, cx, cy
                if len(parts) < 8:
                    raise ValueError(f"PINHOLE model requires 4 parameters, got {len(parts) - 4}")

                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])

                camera = CameraIntrinsics(
                    camera_id=camera_id,
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy
                )

                cameras[camera_id] = camera

            except Exception as e:
                print(f"Warning: Error parsing camera line: {line}")
                print(f"  Error: {str(e)}")
                continue

    print(f"Loaded {len(cameras)} cameras from {filepath.name}")
    return cameras


def load_images(filepath: str) -> Dict[int, CameraPose]:
    """Load camera poses from COLMAP images.txt file.

    Expected format (2 lines per image):
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        # POINTS2D[] as (X, Y, POINT3D_ID)
        1 0.851773 0.0165... 0.503... -0.142... -1.028... 0.0764... 1 frame_0001.jpg
        1.2 3.4 123 5.6 7.8 -1 ...

    Args:
        filepath: Path to images.txt file

    Returns:
        Dictionary mapping image_id to CameraPose

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Images file not found: {filepath}")

    poses = {}
    skip_next = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            # COLMAP images.txt has 2 lines per image
            # Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            # Line 2: POINTS2D (which we skip)
            if skip_next:
                skip_next = False
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                image_id = int(parts[0])
                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                tx = float(parts[5])
                ty = float(parts[6])
                tz = float(parts[7])
                camera_id = int(parts[8])
                image_name = parts[9] if len(parts) > 9 else None

                # Create quaternion and translation vectors
                quaternion = np.array([qw, qx, qy, qz])
                translation = np.array([tx, ty, tz])

                # Validate quaternion is normalized
                q_norm = np.linalg.norm(quaternion)
                if not np.isclose(q_norm, 1.0, atol=1e-3):
                    print(f"Warning: Image {image_id} has non-unit quaternion (norm={q_norm:.6f})")
                    # Normalize it
                    quaternion = quaternion / q_norm

                pose = CameraPose(
                    image_id=image_id,
                    camera_id=camera_id,
                    quaternion=quaternion,
                    translation=translation,
                    image_name=image_name
                )

                poses[image_id] = pose
                skip_next = True  # Skip the POINTS2D line

            except Exception as e:
                print(f"Warning: Error parsing image line: {line}")
                print(f"  Error: {str(e)}")
                continue

    print(f"Loaded {len(poses)} camera poses from {filepath.name}")
    return poses


def get_camera_statistics(cameras: Dict[int, CameraIntrinsics]) -> dict:
    """Compute statistics about camera intrinsics.

    Args:
        cameras: Dictionary of CameraIntrinsics

    Returns:
        Dictionary with statistics
    """
    if not cameras:
        return {}

    fx_values = [cam.fx for cam in cameras.values()]
    fy_values = [cam.fy for cam in cameras.values()]
    cx_values = [cam.cx for cam in cameras.values()]
    cy_values = [cam.cy for cam in cameras.values()]

    stats = {
        'count': len(cameras),
        'fx': {
            'min': np.min(fx_values),
            'max': np.max(fx_values),
            'mean': np.mean(fx_values),
            'std': np.std(fx_values),
        },
        'fy': {
            'min': np.min(fy_values),
            'max': np.max(fy_values),
            'mean': np.mean(fy_values),
            'std': np.std(fy_values),
        },
        'cx': {
            'min': np.min(cx_values),
            'max': np.max(cx_values),
            'mean': np.mean(cx_values),
            'std': np.std(cx_values),
        },
        'cy': {
            'min': np.min(cy_values),
            'max': np.max(cy_values),
            'mean': np.mean(cy_values),
            'std': np.std(cy_values),
        },
    }

    return stats


def get_pose_statistics(poses: Dict[int, CameraPose]) -> dict:
    """Compute statistics about camera poses.

    Args:
        poses: Dictionary of CameraPose

    Returns:
        Dictionary with statistics
    """
    if not poses:
        return {}

    # Extract camera centers
    centers = np.array([pose.C_world for pose in poses.values()])

    # Compute distances from origin
    distances = np.linalg.norm(centers, axis=1)

    # Compute pairwise distances (for scale estimation)
    from scipy.spatial.distance import pdist
    pairwise_dists = pdist(centers)

    stats = {
        'count': len(poses),
        'camera_center_x': {
            'min': np.min(centers[:, 0]),
            'max': np.max(centers[:, 0]),
            'mean': np.mean(centers[:, 0]),
            'std': np.std(centers[:, 0]),
        },
        'camera_center_y': {
            'min': np.min(centers[:, 1]),
            'max': np.max(centers[:, 1]),
            'mean': np.mean(centers[:, 1]),
            'std': np.std(centers[:, 1]),
        },
        'camera_center_z': {
            'min': np.min(centers[:, 2]),
            'max': np.max(centers[:, 2]),
            'mean': np.mean(centers[:, 2]),
            'std': np.std(centers[:, 2]),
        },
        'distance_from_origin': {
            'min': np.min(distances),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'median': np.median(distances),
        },
        'pairwise_distances': {
            'min': np.min(pairwise_dists),
            'max': np.max(pairwise_dists),
            'mean': np.mean(pairwise_dists),
            'median': np.median(pairwise_dists),
        }
    }

    return stats


def validate_camera_pose(pose: CameraPose, intrinsics: CameraIntrinsics) -> bool:
    """Validate camera pose properties.

    Args:
        pose: CameraPose to validate
        intrinsics: CameraIntrinsics for this pose

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check quaternion is normalized
    q_norm = np.linalg.norm(pose.quaternion)
    if not np.isclose(q_norm, 1.0, atol=1e-3):
        raise ValueError(f"Quaternion not normalized: ||q|| = {q_norm}")

    # Check rotation matrix properties
    R = pose.R
    det_R = np.linalg.det(R)
    if not np.isclose(det_R, 1.0, atol=1e-3):
        raise ValueError(f"Rotation matrix has det(R) = {det_R}, expected 1.0")

    # Check orthogonality
    I = R @ R.T
    if not np.allclose(I, np.eye(3), atol=1e-3):
        raise ValueError("Rotation matrix is not orthogonal")

    # Check camera center is finite
    C = pose.C_world
    if not np.all(np.isfinite(C)):
        raise ValueError(f"Camera center has non-finite values: {C}")

    return True
