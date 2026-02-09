"""
Ray casting utilities for unprojecting 2D pixels to 3D rays.
"""

import numpy as np
from typing import Tuple

from .data_structures import Ray3D, CameraIntrinsics, CameraPose


def unproject_pixel(
    pixel_x: float,
    pixel_y: float,
    camera_intrinsics: CameraIntrinsics,
    camera_pose: CameraPose
) -> Ray3D:
    """Unproject a 2D pixel coordinate to a 3D ray in world space.

    This implements the full unprojection pipeline:
    1. Convert pixel coords to normalized image coords
    2. Create direction vector in camera space
    3. Transform to world space using camera pose
    4. Create ray from camera center through the point

    Args:
        pixel_x: X coordinate in image (pixels)
        pixel_y: Y coordinate in image (pixels)
        camera_intrinsics: Camera intrinsic parameters (K matrix)
        camera_pose: Camera extrinsic parameters (R, t)

    Returns:
        Ray3D object with origin at camera center and direction in world space

    Mathematical Background:
        Pixel to normalized coords:
            x_norm = (pixel_x - cx) / fx
            y_norm = (pixel_y - cy) / fy

        Direction in camera space:
            dir_cam = [x_norm, y_norm, 1.0]

        Transform to world space:
            dir_world = R^T @ dir_cam  (R is world-to-camera)

        Ray equation:
            P(t) = C_world + t * dir_world
    """
    # Get intrinsic parameters
    fx = camera_intrinsics.fx
    fy = camera_intrinsics.fy
    cx = camera_intrinsics.cx
    cy = camera_intrinsics.cy

    # Step 1: Normalize pixel to image plane
    x_norm = (pixel_x - cx) / fx
    y_norm = (pixel_y - cy) / fy

    # Step 2: Create direction vector in camera space
    # In camera space, Z=1 is the image plane at focal length distance
    direction_camera = np.array([x_norm, y_norm, 1.0])

    # Step 3: Transform direction to world space
    # R is world-to-camera, so R^T transforms camera-to-world
    R = camera_pose.R
    direction_world = R.T @ direction_camera

    # Step 4: Get camera center in world coordinates
    camera_center = camera_pose.C_world

    # Step 5: Create and return ray
    # Ray3D constructor automatically normalizes the direction
    ray = Ray3D(origin=camera_center, direction=direction_world)

    return ray


def unproject_bbox_bottom_center(
    bbox,
    camera_intrinsics: CameraIntrinsics,
    camera_pose: CameraPose
) -> Ray3D:
    """Unproject the bottom-center point of a bounding box to a 3D ray.

    This is a convenience function that combines bbox.bottom_center with
    unproject_pixel.

    Args:
        bbox: BoundingBox object
        camera_intrinsics: Camera intrinsic parameters
        camera_pose: Camera extrinsic parameters

    Returns:
        Ray3D from camera center through bbox bottom-center point
    """
    # Get bottom-center pixel coordinates
    pixel_x, pixel_y = bbox.bottom_center

    # Unproject to 3D ray
    return unproject_pixel(pixel_x, pixel_y, camera_intrinsics, camera_pose)


def validate_unprojection(
    point_3d: np.ndarray,
    camera_intrinsics: CameraIntrinsics,
    camera_pose: CameraPose,
    expected_pixel: Tuple[float, float],
    tolerance: float = 1.0
) -> bool:
    """Validate unprojection by projecting a 3D point back to 2D.

    This is useful for testing: project a known 3D point to 2D, then
    verify it matches the expected pixel coordinates.

    Args:
        point_3d: 3D point in world coordinates (3,)
        camera_intrinsics: Camera intrinsic parameters
        camera_pose: Camera extrinsic parameters
        expected_pixel: Expected (x, y) pixel coordinates
        tolerance: Maximum allowed pixel error

    Returns:
        True if reprojection error is within tolerance

    Raises:
        ValueError: If reprojection error exceeds tolerance
    """
    # Get camera parameters
    K = camera_intrinsics.K
    R = camera_pose.R
    t = camera_pose.translation

    # Transform 3D point to camera space
    # p_cam = R @ p_world + t
    point_camera = R @ point_3d + t

    # Check that point is in front of camera
    if point_camera[2] <= 0:
        raise ValueError(
            f"Point is behind camera (z={point_camera[2]:.3f}). "
            "Cannot project to image."
        )

    # Project to image plane
    # p_image = K @ p_cam / p_cam[2]
    point_homo = K @ point_camera
    pixel_x = point_homo[0] / point_homo[2]
    pixel_y = point_homo[1] / point_homo[2]

    # Compute reprojection error
    error_x = pixel_x - expected_pixel[0]
    error_y = pixel_y - expected_pixel[1]
    error = np.sqrt(error_x**2 + error_y**2)

    if error > tolerance:
        raise ValueError(
            f"Reprojection error {error:.2f} px exceeds tolerance {tolerance} px.\n"
            f"  Expected: ({expected_pixel[0]:.1f}, {expected_pixel[1]:.1f})\n"
            f"  Got:      ({pixel_x:.1f}, {pixel_y:.1f})"
        )

    return True


def project_3d_to_2d(
    point_3d: np.ndarray,
    camera_intrinsics: CameraIntrinsics,
    camera_pose: CameraPose
) -> Tuple[float, float]:
    """Project a 3D point in world space to 2D pixel coordinates.

    This is the forward projection (opposite of unprojection).
    Useful for visualizing 3D intersection points on video frames.

    Args:
        point_3d: 3D point in world coordinates (3,)
        camera_intrinsics: Camera intrinsic parameters
        camera_pose: Camera extrinsic parameters

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates

    Raises:
        ValueError: If point is behind camera
    """
    # Get camera parameters
    K = camera_intrinsics.K
    R = camera_pose.R
    t = camera_pose.translation

    # Transform to camera space
    point_camera = R @ point_3d + t

    # Check point is in front of camera
    if point_camera[2] <= 0:
        raise ValueError(
            f"Point is behind camera (z={point_camera[2]:.3f}). "
            "Cannot project to image."
        )

    # Project to image plane
    point_homo = K @ point_camera
    pixel_x = point_homo[0] / point_homo[2]
    pixel_y = point_homo[1] / point_homo[2]

    return (pixel_x, pixel_y)


def compute_ray_direction_angles(ray: Ray3D) -> dict:
    """Compute angles of ray direction for debugging/visualization.

    Args:
        ray: Ray3D object

    Returns:
        Dictionary with angle information:
            - azimuth: Horizontal angle (degrees, 0=+X, 90=+Y)
            - elevation: Vertical angle (degrees, 0=horizontal, 90=up)
            - direction_unit: Unit direction vector
    """
    direction = ray.direction

    # Azimuth (horizontal angle in XY plane)
    azimuth_rad = np.arctan2(direction[1], direction[0])
    azimuth_deg = np.degrees(azimuth_rad)

    # Elevation (angle from horizontal plane)
    horizontal_dist = np.sqrt(direction[0]**2 + direction[1]**2)
    elevation_rad = np.arctan2(direction[2], horizontal_dist)
    elevation_deg = np.degrees(elevation_rad)

    return {
        'azimuth_deg': azimuth_deg,
        'elevation_deg': elevation_deg,
        'direction_unit': direction,
    }
