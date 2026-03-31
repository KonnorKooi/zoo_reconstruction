"""Visualize camera positions and ray-mesh intersections in 3D.

For the pool and/or yard cameras, casts a ray through the bottom-center of a
bounding box from a chosen frame, intersects it with the Poisson mesh, and
exports a .glb scene file containing:
  - The mesh (grey)
  - Camera positions (spheres)
  - Rays (lines from camera to intersection)
  - Ground intersection dots (large spheres)

Open the output .glb in MeshLab, Blender, or drag it into
  https://gltf-viewer.donmccurdy.com

Usage:
    uv run python visualize_cameras.py --camera both --frame 0
    uv run python visualize_cameras.py --camera pool --frame 50
    uv run python visualize_cameras.py --camera both --frame 100 --out scene.glb
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import trimesh.creation
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPARSE_DIR = Path("/home/konnorkooi/code/research/best_recon/registered")
MESH_PATH  = Path("/home/konnorkooi/code/research/best_recon/dense/meshed-poisson.ply")

_HERE = Path(__file__).parent
BBOX_POOL = _HERE / "rhino" / "rhino_pool_1_trimmed_bbox.txt"
BBOX_YARD = _HERE / "rhino" / "rhino_yard_1_bbox.txt"

# Bounding boxes were detected on 1920x1080 video; COLMAP images are 1009x669.
VIDEO_W, VIDEO_H   = 1920, 1080
COLMAP_W, COLMAP_H = 1009,  669
SCALE_X = COLMAP_W / VIDEO_W
SCALE_Y = COLMAP_H / VIDEO_H


# ---------------------------------------------------------------------------
# Camera loading (supports SIMPLE_PINHOLE and PINHOLE)
# ---------------------------------------------------------------------------
def load_cameras(filepath):
    cameras = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            width  = int(parts[2])
            height = int(parts[3])
            if model == 'SIMPLE_PINHOLE':
                f_  = float(parts[4])
                cx  = float(parts[5])
                cy  = float(parts[6])
                cameras[cam_id] = dict(width=width, height=height,
                                       fx=f_, fy=f_, cx=cx, cy=cy)
            elif model == 'PINHOLE':
                fx  = float(parts[4])
                fy  = float(parts[5])
                cx  = float(parts[6])
                cy  = float(parts[7])
                cameras[cam_id] = dict(width=width, height=height,
                                       fx=fx, fy=fy, cx=cx, cy=cy)
            # other models ignored (not needed here)
    return cameras


def load_poses(filepath, target_names):
    """Return pose dict keyed by image name, for the requested filenames."""
    poses = {}
    skip_next = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if skip_next:
                skip_next = False
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            image_name = parts[9]
            if image_name in target_names:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz     = map(float, parts[5:8])
                poses[image_name] = dict(
                    camera_id=int(parts[8]),
                    q=np.array([qw, qx, qy, qz]),
                    t=np.array([tx, ty, tz]),
                )
            skip_next = True
    return poses


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def quat_to_R(q):
    qw, qx, qy, qz = q
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def camera_center(q, t):
    R = quat_to_R(q)
    return -R.T @ t


def unproject_ray(px, py, intr, q, t):
    """Return (origin, unit_direction) in world space."""
    x_norm = (px - intr['cx']) / intr['fx']
    y_norm = (py - intr['cy']) / intr['fy']
    dir_cam   = np.array([x_norm, y_norm, 1.0])
    R         = quat_to_R(q)
    dir_world = R.T @ dir_cam
    dir_world /= np.linalg.norm(dir_world)
    return camera_center(q, t), dir_world


def load_bbox(filepath, frame_idx):
    lines = [l.strip() for l in open(filepath) if l.strip()]
    if frame_idx >= len(lines):
        raise IndexError(f"Frame {frame_idx} out of range (file has {len(lines)} frames)")
    x, y, w, h = map(float, lines[frame_idx].split(','))
    return x, y, w, h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize ray-mesh intersections")
    parser.add_argument('--camera', choices=['pool', 'yard', 'both'], default='both',
                        help='Which camera(s) to show')
    parser.add_argument('--frame', type=int, default=0,
                        help='Bbox frame index to use for each camera')
    parser.add_argument('--out', default='scene.glb',
                        help='Output .glb filename (default: scene.glb)')
    args = parser.parse_args()

    # Load COLMAP data
    cameras = load_cameras(SPARSE_DIR / "cameras.txt")
    poses   = load_poses(SPARSE_DIR / "images.txt",
                         {'rhino_pool_frame.jpg', 'rhino_yard_1.jpg'})
    print(f"Loaded poses for: {list(poses.keys())}")

    # Load mesh
    print(f"Loading mesh from {MESH_PATH} ...")
    scene_mesh = trimesh.load(str(MESH_PATH), force='mesh')
    print(f"  {len(scene_mesh.vertices):,} vertices, {len(scene_mesh.faces):,} faces")
    # Paint mesh neutral grey
    scene_mesh.visual.vertex_colors = np.full(
        (len(scene_mesh.vertices), 4), [160, 160, 160, 220], dtype=np.uint8
    )

    # Build trimesh raycasting intersector
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(scene_mesh)

    # Assemble trimesh scene
    ts = trimesh.Scene()
    ts.add_geometry(scene_mesh, node_name='mesh')

    configs = []
    if args.camera in ('pool', 'both'):
        configs.append(('rhino_pool_frame.jpg', BBOX_POOL, [230,  50,  50, 255]))  # red
    if args.camera in ('yard', 'both'):
        configs.append(('rhino_yard_1.jpg',     BBOX_YARD, [ 50, 130, 255, 255]))  # blue

    for img_name, bbox_file, rgba in configs:
        if img_name not in poses:
            print(f"WARNING: pose for '{img_name}' not found — skipping")
            continue

        pose  = poses[img_name]
        intr  = cameras[pose['camera_id']]
        q, t  = pose['q'], pose['t']
        C     = camera_center(q, t)
        color_rgb = [c / 255.0 for c in rgba[:3]]

        # Load bbox and scale from 1920x1080 → COLMAP image space
        bx, by, bw, bh = load_bbox(bbox_file, args.frame)
        bx_s = bx * SCALE_X;  bw_s = bw * SCALE_X
        by_s = by * SCALE_Y;  bh_s = bh * SCALE_Y
        px = bx_s + bw_s / 2   # bottom-center x
        py = by_s + bh_s        # bottom-center y

        print(f"\n{img_name}  (cam_id={pose['camera_id']})")
        print(f"  Camera center world: [{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}]")
        print(f"  Bbox (1920x1080):    x={bx:.0f} y={by:.0f} w={bw:.0f} h={bh:.0f}")
        print(f"  Bottom-center pixel: ({px:.1f}, {py:.1f})  [COLMAP space]")

        origin, direction = unproject_ray(px, py, intr, q, t)

        # Raycast with trimesh
        locs, _idx_ray, _idx_tri = intersector.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=False,
        )

        if len(locs) > 0:
            # Pick closest hit to ray origin
            dists = np.linalg.norm(locs - origin, axis=1)
            intersection = locs[np.argmin(dists)]
            t_hit = float(np.min(dists))
            print(f"  Intersection:        [{intersection[0]:.3f}, {intersection[1]:.3f}, {intersection[2]:.3f}]  (t={t_hit:.3f})")
            ray_end = intersection
        else:
            print(f"  No intersection found — drawing ray 15 units long")
            ray_end = origin + direction * 15.0
            intersection = None

        tag = img_name.split('.')[0]

        # Camera sphere
        cam_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.15)
        cam_sphere.apply_translation(C)
        cam_sphere.visual.vertex_colors = rgba
        ts.add_geometry(cam_sphere, node_name=f'{tag}_camera')

        # Ray as a thin cylinder
        ray_vec  = ray_end - origin
        ray_len  = float(np.linalg.norm(ray_vec))
        ray_dir  = ray_vec / ray_len
        # Build cylinder along Z then rotate to ray direction
        cyl = trimesh.creation.cylinder(radius=0.02, height=ray_len, sections=8)
        # Cylinder default axis is Z; rotate to ray_dir
        z_axis = np.array([0.0, 0.0, 1.0])
        rot_axis = np.cross(z_axis, ray_dir)
        rot_norm = np.linalg.norm(rot_axis)
        if rot_norm > 1e-6:
            rot_axis /= rot_norm
            angle = float(np.arccos(np.clip(np.dot(z_axis, ray_dir), -1, 1)))
            R_ray = trimesh.transformations.rotation_matrix(angle, rot_axis)
        else:
            R_ray = np.eye(4)
        # Translate so base sits at origin, then move midpoint along ray
        midpoint = origin + ray_dir * ray_len / 2.0
        T_ray = trimesh.transformations.translation_matrix(midpoint)
        cyl.apply_transform(T_ray @ R_ray)
        cyl.visual.vertex_colors = rgba
        ts.add_geometry(cyl, node_name=f'{tag}_ray')

        # Ground intersection dot
        if intersection is not None:
            dot = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
            dot.apply_translation(intersection)
            dot.visual.vertex_colors = rgba
            ts.add_geometry(dot, node_name=f'{tag}_dot')

    out_path = Path(args.out)
    ts.export(str(out_path))
    print(f"\nScene exported → {out_path.resolve()}")
    print("Open with:  MeshLab, Blender, or drag into https://gltf-viewer.donmccurdy.com")


if __name__ == '__main__':
    main()
