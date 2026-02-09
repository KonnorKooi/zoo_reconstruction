"""
Data structures for rhino tracking and 3D reconstruction.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class BoundingBox:
    """Represents a 2D bounding box in image coordinates.

    Attributes:
        frame_id: Frame number (0-indexed)
        x: Top-left x coordinate in pixels
        y: Top-left y coordinate in pixels
        width: Width in pixels
        height: Height in pixels
    """
    frame_id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Returns (x, y) pixel coordinates of bottom-middle point.

        This is the point we'll use for ray casting - the rhino's
        ground contact point.
        """
        return (self.x + self.width / 2, self.y + self.height)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns (x, y) pixel coordinates of bbox center."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def top_left(self) -> Tuple[float, float]:
        """Returns (x, y) of top-left corner."""
        return (self.x, self.y)

    @property
    def bottom_right(self) -> Tuple[float, float]:
        """Returns (x, y) of bottom-right corner."""
        return (self.x + self.width, self.y + self.height)

    def __repr__(self):
        return f"BBox(frame={self.frame_id}, x={self.x:.1f}, y={self.y:.1f}, w={self.width:.1f}, h={self.height:.1f})"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters (PINHOLE model).

    Attributes:
        camera_id: Unique camera identifier
        width: Image width in pixels
        height: Image height in pixels
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)
        cx: Principal point x offset (pixels)
        cy: Principal point y offset (pixels)
    """
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        """Returns 3x3 camera intrinsic matrix.

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        return np.array([
            [self.fx,      0, self.cx],
            [     0, self.fy, self.cy],
            [     0,      0,       1]
        ])

    def __repr__(self):
        return f"CameraIntrinsics(id={self.camera_id}, size=({self.width}x{self.height}), f=({self.fx:.1f},{self.fy:.1f}))"


@dataclass
class CameraPose:
    """Camera extrinsic parameters (pose in world coordinates).

    COLMAP convention: Stores world-to-camera transformation.

    Attributes:
        image_id: Unique image identifier
        camera_id: Reference to CameraIntrinsics
        quaternion: Rotation as [qw, qx, qy, qz] (unit quaternion)
        translation: Translation vector [tx, ty, tz]
        image_name: Optional image filename
    """
    image_id: int
    camera_id: int
    quaternion: np.ndarray  # [qw, qx, qy, qz]
    translation: np.ndarray  # [tx, ty, tz]
    image_name: Optional[str] = None

    @property
    def R(self) -> np.ndarray:
        """Returns 3x3 rotation matrix (world-to-camera).

        Uses scipy for robust quaternion conversion.
        """
        from scipy.spatial.transform import Rotation
        # scipy expects [qx, qy, qz, qw] order
        qw, qx, qy, qz = self.quaternion
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        return R

    @property
    def C_world(self) -> np.ndarray:
        """Returns camera center in world coordinates.

        Since COLMAP stores world-to-camera as:
            p_cam = R @ p_world + t

        Camera center is at origin in camera space:
            0 = R @ C + t
            C = -R^T @ t
        """
        R = self.R
        return -R.T @ self.translation

    @property
    def R_inv(self) -> np.ndarray:
        """Returns inverse rotation (camera-to-world).

        Since R is orthogonal: R^-1 = R^T
        """
        return self.R.T

    def __repr__(self):
        return f"CameraPose(img_id={self.image_id}, cam_id={self.camera_id}, C={self.C_world})"


@dataclass
class Ray3D:
    """Represents a 3D ray in world coordinates.

    Ray equation: P(t) = origin + t * direction, where t >= 0

    Attributes:
        origin: Ray origin point (3D coordinates)
        direction: Ray direction (normalized unit vector)
    """
    origin: np.ndarray  # (3,)
    direction: np.ndarray  # (3,) - should be unit vector

    def __post_init__(self):
        """Validate ray properties."""
        # Ensure numpy arrays
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)

        # Validate shapes
        assert self.origin.shape == (3,), f"Origin must be 3D, got shape {self.origin.shape}"
        assert self.direction.shape == (3,), f"Direction must be 3D, got shape {self.direction.shape}"

        # Normalize direction (ensure unit vector)
        norm = np.linalg.norm(self.direction)
        if norm < 1e-10:
            raise ValueError(f"Direction vector has near-zero norm: {norm}")
        self.direction = self.direction / norm

    def point_at(self, t: float) -> np.ndarray:
        """Returns point along ray at parameter t.

        Args:
            t: Distance parameter (should be >= 0)

        Returns:
            3D point: origin + t * direction
        """
        return self.origin + t * self.direction

    def __repr__(self):
        return f"Ray3D(origin={self.origin}, dir={self.direction})"
