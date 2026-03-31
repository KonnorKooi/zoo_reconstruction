# Rhino 3D Tracking

Track a rhino's 3D position in video by casting rays from 2D bounding box detections through a calibrated camera into a 3D mesh reconstruction of the enclosure.

## How It Works

1. A 2D bounding box detector (run separately) produces per-frame bounding boxes in CSV format.
2. For each frame, the bottom-center of the bounding box is unprojected into a 3D ray using the known camera intrinsics and pose (from COLMAP).
3. That ray is intersected with a dense mesh of the enclosure (PLY format, from COLMAP dense reconstruction).
4. The intersection point is the estimated ground-contact position of the rhino in 3D.

The camera poses and intrinsics come from COLMAP sparse reconstruction. The mesh comes from COLMAP dense reconstruction (Poisson surface). Both are produced by the pipeline in `../colmap/manual_registration/`.

## Directory Structure

```
rhino-tracking/
├── src/                         # Core library
│   ├── data_structures.py       # BoundingBox, CameraIntrinsics, CameraPose, Ray3D
│   ├── camera_parser.py         # Load COLMAP cameras.txt and images.txt
│   ├── bbox_parser.py           # Load bounding boxes from CSV
│   ├── ray_casting.py           # Unproject pixels to 3D rays, reproject 3D to 2D
│   └── mesh_intersection.py     # Ray-mesh intersection via trimesh
├── extract_video_frames.py      # Extract keyframes from video at regular intervals
├── live_track.py                # Live 3D visualization using Rerun
├── resize_rhino_frames.py       # Resize extracted frames
├── check_image_sizes.py         # Verify frame dimensions
├── visualize_cameras.py         # Visualize camera frustums in 3D
├── rhino/                       # Bounding box CSVs and video data
├── model-56a-southern-white-rhino/  # Rhino 3D model for visualization
├── scene.glb                    # Scene file for Rerun visualization
└── pyproject.toml
```

## Setup

Requires Python 3.9--3.12.

```bash
cd rhino-tracking
uv sync
```

Optional extras:

```bash
uv sync --extra colmap   # pycolmap bindings
uv sync --extra viz      # pyvista 3D viewer
```

## Usage

### 1. Extract video frames

Pull frames from a stationary camera video at a regular interval (default: every 30th frame).

```bash
uv run extract_video_frames.py --video path/to/video.mp4 --output frames/ --interval 30
```

### 2. Run the tracking pipeline

`live_track.py` loads the COLMAP model, mesh, and bounding boxes, pre-computes all ray-mesh intersections, and streams the results to a Rerun viewer in the browser.

```bash
uv run live_track.py
```

Paths to the COLMAP sparse model, dense mesh, bounding box CSV, and video frames are configured at the top of the script. Update these to point at your data before running.

### 3. Inspect cameras (optional)

Visualize camera positions and orientations from the COLMAP model.

```bash
uv run visualize_cameras.py
```

## Core Library (src/)

### data_structures.py

- **BoundingBox** -- 2D detection box with a `bottom_center` property (the pixel used as the ground-contact point).
- **CameraIntrinsics** -- PINHOLE model: fx, fy, cx, cy.
- **CameraPose** -- COLMAP world-to-camera transform (quaternion + translation). Camera center in world coordinates: `C = -R^T @ t`.
- **Ray3D** -- 3D ray with origin and normalized direction.

### camera_parser.py

Loads COLMAP text-format `cameras.txt` and `images.txt`. Validates quaternions and rotation matrices. Computes statistics on intrinsics and poses.

### bbox_parser.py

Loads 2D bounding boxes from CSV (one row per frame). Validates that boxes fall within image bounds. Includes `plot_bbox_on_frame()` for debugging overlays.

### ray_casting.py

- `unproject_pixel(px, py, K, pose)` -- converts a 2D pixel to a world-space Ray3D.
  - Normalizes pixel coordinates via K inverse.
  - Rotates direction from camera space to world space via R transpose.
  - Anchors the ray at the camera center.
- `unproject_bbox_bottom_center(bbox, K, pose)` -- convenience wrapper.
- `project_3d_to_2d(point, K, pose)` -- inverse operation for visualization.

### mesh_intersection.py

`MeshIntersector` class wrapping trimesh:
- Loads PLY meshes, validates geometry (removes degenerate faces).
- `intersect_ray(ray)` -- single ray intersection, returns closest hit.
- `intersect_rays_batch(rays)` -- batched intersection.
- `get_intersection_info(ray)` -- returns hit point, distance, and surface normal.
- Visualization helpers: `create_sphere_at_point()`, `visualize_ray_mesh_intersection()`.

## Coordinate System

COLMAP convention: world-to-camera (W2C), X-right, Y-down, Z-forward.

Camera center in world coordinates:
```
C_world = -R^T @ t
```

Ray direction from a pixel `(u, v)`:
```
d_cam = K^{-1} @ [u, v, 1]^T
d_world = R^T @ d_cam
```

## Dependencies

Core: numpy, scipy, opencv-python
3D: trimesh, open3d, rtree, fast-simplification
Visualization: rerun-sdk, matplotlib, jupyter
