"""
Bounding box parser for rhino tracking data.
"""

from typing import List
import numpy as np
from pathlib import Path

from .data_structures import BoundingBox


def load_bboxes(filepath: str) -> List[BoundingBox]:
    """Load bounding boxes from text file.

    Expected format: One bbox per line in CSV format
        x,y,width,height

    Each line corresponds to one frame (0-indexed).

    Args:
        filepath: Path to bbox text file

    Returns:
        List of BoundingBox objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Bbox file not found: {filepath}")

    bboxes = []
    with open(filepath, 'r') as f:
        for frame_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split(',')
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 values, got {len(parts)}")

                x, y, w, h = map(float, parts)

                # Basic validation
                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid bbox dimensions: w={w}, h={h}")

                bbox = BoundingBox(
                    frame_id=frame_id,
                    x=x,
                    y=y,
                    width=w,
                    height=h
                )
                bboxes.append(bbox)

            except Exception as e:
                raise ValueError(f"Error parsing line {frame_id + 1}: {line}\n{str(e)}")

    print(f"Loaded {len(bboxes)} bounding boxes from {filepath.name}")
    return bboxes


def validate_bboxes(bboxes: List[BoundingBox], image_width: int, image_height: int) -> bool:
    """Validate that bboxes are within image bounds.

    Args:
        bboxes: List of BoundingBox objects
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        True if all bboxes are valid

    Raises:
        ValueError: If any bbox is out of bounds
    """
    for bbox in bboxes:
        # Check top-left corner
        if bbox.x < 0 or bbox.x >= image_width:
            raise ValueError(
                f"Frame {bbox.frame_id}: x={bbox.x} out of bounds [0, {image_width})"
            )
        if bbox.y < 0 or bbox.y >= image_height:
            raise ValueError(
                f"Frame {bbox.frame_id}: y={bbox.y} out of bounds [0, {image_height})"
            )

        # Check bottom-right corner
        br_x, br_y = bbox.bottom_right
        if br_x > image_width or br_y > image_height:
            raise ValueError(
                f"Frame {bbox.frame_id}: bottom-right ({br_x}, {br_y}) "
                f"exceeds image bounds ({image_width}, {image_height})"
            )

    print(f"✓ All {len(bboxes)} bboxes are within image bounds ({image_width}x{image_height})")
    return True


def get_bbox_statistics(bboxes: List[BoundingBox]) -> dict:
    """Compute statistics about bounding boxes.

    Args:
        bboxes: List of BoundingBox objects

    Returns:
        Dictionary with statistics
    """
    if not bboxes:
        return {}

    widths = [b.width for b in bboxes]
    heights = [b.height for b in bboxes]
    areas = [b.width * b.height for b in bboxes]
    centers_x = [b.center[0] for b in bboxes]
    centers_y = [b.center[1] for b in bboxes]

    stats = {
        'count': len(bboxes),
        'width': {
            'min': np.min(widths),
            'max': np.max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths),
        },
        'height': {
            'min': np.min(heights),
            'max': np.max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights),
        },
        'area': {
            'min': np.min(areas),
            'max': np.max(areas),
            'mean': np.mean(areas),
            'std': np.std(areas),
        },
        'center_x': {
            'min': np.min(centers_x),
            'max': np.max(centers_x),
            'mean': np.mean(centers_x),
            'std': np.std(centers_x),
        },
        'center_y': {
            'min': np.min(centers_y),
            'max': np.max(centers_y),
            'mean': np.mean(centers_y),
            'std': np.std(centers_y),
        },
    }

    return stats


def plot_bbox_on_frame(frame: np.ndarray, bbox: BoundingBox, show_points: bool = True):
    """Plot bounding box on video frame.

    Args:
        frame: Image as numpy array (H, W, 3)
        bbox: BoundingBox object
        show_points: If True, also show center and bottom-center points

    Returns:
        Image with bbox drawn
    """
    import cv2

    # Make a copy to avoid modifying original
    img = frame.copy()

    # Draw bbox rectangle (blue)
    x, y = int(bbox.x), int(bbox.y)
    x2, y2 = int(bbox.x + bbox.width), int(bbox.y + bbox.height)
    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

    if show_points:
        # Draw center point (green)
        cx, cy = bbox.center
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        # Draw bottom-center point (red) - this is what we'll use for ray casting
        bcx, bcy = bbox.bottom_center
        cv2.circle(img, (int(bcx), int(bcy)), 7, (0, 0, 255), -1)

        # Add labels
        cv2.putText(img, "center", (int(cx) + 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "bottom", (int(bcx) + 10, int(bcy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Add frame number
    cv2.putText(img, f"Frame {bbox.frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img
