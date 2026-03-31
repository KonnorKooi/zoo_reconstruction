"""Live rhino tracking visualization with rerun.

Pre-computes all ray-mesh intersections in one batch, then streams the results
to rerun. Opens in your browser — use the timeline scrubber to play through
frames and watch the 3D ground position move as the rhino walks.

Usage:
    uv run python live_track.py
    uv run python live_track.py --camera pool
    uv run python live_track.py --every 5    # subsample every 5th frame (faster load)
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPARSE_DIR  = Path("/home/konnorkooi/code/research/best_recon/registered")
MESH_PATH   = Path("/home/konnorkooi/code/research/best_recon/dense/meshed-poisson.ply")
POINTS_PATH = Path("/home/konnorkooi/code/research/best_recon/dense/fused.ply")

_HERE      = Path(__file__).parent
BBOX_POOL  = _HERE / "rhino" / "rhino_pool_1_trimmed_bbox.txt"
RHINO_MODEL = _HERE / "model-56a-southern-white-rhino" / "source" / "rhinoThird_twoSevenNine_test.glb"
BBOX_YARD = _HERE / "rhino" / "rhino_yard_1_bbox.txt"

VIDEO_W, VIDEO_H   = 1920, 1080
COLMAP_W, COLMAP_H = 1009,  669
SCALE_X = COLMAP_W / VIDEO_W
SCALE_Y = COLMAP_H / VIDEO_H


# ---------------------------------------------------------------------------
# Camera / pose helpers
# ---------------------------------------------------------------------------
def load_cameras(filepath):
    cameras = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            cam_id, model = int(parts[0]), parts[1]
            width, height = int(parts[2]), int(parts[3])
            if model == 'SIMPLE_PINHOLE':
                f_ = float(parts[4]); cx = float(parts[5]); cy = float(parts[6])
                cameras[cam_id] = dict(width=width, height=height, fx=f_, fy=f_, cx=cx, cy=cy)
            elif model == 'PINHOLE':
                fx = float(parts[4]); fy = float(parts[5])
                cx = float(parts[6]); cy = float(parts[7])
                cameras[cam_id] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


def load_poses(filepath, target_names):
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
            name = parts[9]
            if name in target_names:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz     = map(float, parts[5:8])
                poses[name] = dict(camera_id=int(parts[8]),
                                   q=np.array([qw, qx, qy, qz]),
                                   t=np.array([tx, ty, tz]))
            skip_next = True
    return poses


def quat_to_R(q):
    qw, qx, qy, qz = q
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

def camera_center(q, t):
    return -quat_to_R(q).T @ t

def unproject_rays(pixels_xy, intr, q, t):
    """Unproject (N,2) pixel array to (N,3) world-space ray directions + shared origin."""
    origin = camera_center(q, t)
    R      = quat_to_R(q)
    x_n    = (pixels_xy[:, 0] - intr['cx']) / intr['fx']
    y_n    = (pixels_xy[:, 1] - intr['cy']) / intr['fy']
    ones   = np.ones(len(pixels_xy))
    dirs_cam   = np.stack([x_n, y_n, ones], axis=1)   # (N, 3)
    dirs_world = dirs_cam @ R                           # R^T each row = (R @ row^T)^T
    norms      = np.linalg.norm(dirs_world, axis=1, keepdims=True)
    dirs_world /= norms
    return origin, dirs_world

def load_bboxes(filepath):
    lines = [l.strip() for l in open(filepath) if l.strip()]
    return np.array([list(map(float, l.split(','))) for l in lines])  # (N,4) x,y,w,h


# ---------------------------------------------------------------------------
# Batch raycast: returns (N,3) intersections, NaN where no hit
# ---------------------------------------------------------------------------
def batch_raycast(intersector, origins, directions):
    """origins: (N,3) or (3,) broadcast, directions: (N,3). Returns (N,3) or NaN.

    Takes the LAST (furthest) intersection per ray so that rays passing through
    obstacles land on the surface behind them rather than the obstacle face.
    If a ray misses entirely, returns NaN for that entry.
    """
    N = len(directions)
    if origins.ndim == 1:
        origins = np.tile(origins, (N, 1))

    locs, idx_ray, _ = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,   # get all hits so we can pick the last one
    )

    results = np.full((N, 3), np.nan)
    if len(locs) > 0:
        # For each hit keep the FURTHEST intersection (last surface the ray passed through)
        for hit_i, ray_i in enumerate(idx_ray):
            pt   = locs[hit_i]
            dist = np.linalg.norm(pt - origins[ray_i])
            existing = results[ray_i]
            if np.any(np.isnan(existing)) or dist > np.linalg.norm(existing - origins[ray_i]):
                results[ray_i] = pt

    # Temporal fallback: propagate last known position to frames that missed entirely
    last_known = None
    for i in range(N):
        if not np.any(np.isnan(results[i])):
            last_known = results[i].copy()
        elif last_known is not None:
            results[i] = last_known

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', choices=['pool', 'yard', 'both'], default='both')
    parser.add_argument('--every',  type=int, default=1,
                        help='Use every Nth frame (default 1 = all)')
    parser.add_argument('--rhino-scale', type=float, default=0.15,
                        help='Scale factor for the rhino model (default 1.0)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load everything before touching rerun
    # ------------------------------------------------------------------
    cameras = load_cameras(SPARSE_DIR / "cameras.txt")
    poses   = load_poses(SPARSE_DIR / "images.txt",
                         {'rhino_pool_frame.jpg', 'rhino_yard_1.jpg'})

    print("Loading point cloud (colored)...")
    pcloud = trimesh.load(str(POINTS_PATH))
    # Subsample for display — 950k points is too heavy for rerun
    step = 4
    pc_verts  = np.array(pcloud.vertices[::step],                    dtype=np.float32)
    pc_colors = np.array(pcloud.visual.vertex_colors[::step, :3],    dtype=np.uint8)
    print(f"  Subsampled to {len(pc_verts):,} points (1 in {step})")
    del pcloud

    print("Loading mesh (for raycasting)...")
    mesh = trimesh.load(str(MESH_PATH), force='mesh')
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Decimate heavily — ground intersection doesn't need 3.5M faces
    target_reduction = 1.0 - (80_000 / len(mesh.faces))
    print(f"  Decimating by {target_reduction:.2%}...")
    mesh = mesh.simplify_quadric_decimation(target_reduction)
    print(f"  Decimated: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    del mesh

    # ------------------------------------------------------------------
    # 2. Pre-compute all intersections (batch, fast)
    # ------------------------------------------------------------------
    configs = []
    if args.camera in ('pool', 'both'):
        configs.append(('rhino_pool_frame.jpg', BBOX_POOL, 'pool', [230, 60,  60]))
    if args.camera in ('yard', 'both'):
        configs.append(('rhino_yard_1.jpg',     BBOX_YARD, 'yard', [ 60, 130, 255]))

    camera_data = {}   # tag → {origin, directions, intersections, color, C}
    for img_name, bbox_file, tag, color in configs:
        if img_name not in poses:
            print(f"WARNING: no pose for {img_name}, skipping")
            continue
        pose  = poses[img_name]
        intr  = cameras[pose['camera_id']]
        q, t  = pose['q'], pose['t']
        C     = camera_center(q, t)

        bboxes = load_bboxes(bbox_file)
        if args.every > 1:
            bboxes = bboxes[::args.every]
        N = len(bboxes)

        # Bottom-center pixels, scaled to COLMAP space
        px = (bboxes[:, 0] + bboxes[:, 2] / 2) * SCALE_X
        py = (bboxes[:, 1] + bboxes[:, 3])      * SCALE_Y
        pixels = np.stack([px, py], axis=1)

        print(f"Computing {N} rays for {tag}...")
        origin, directions = unproject_rays(pixels, intr, q, t)
        origins_arr = np.tile(origin, (N, 1))

        print(f"Raycasting {N} rays against mesh (this may take a minute)...")
        intersections = batch_raycast(intersector, origins_arr, directions)
        hits = np.sum(~np.any(np.isnan(intersections), axis=1))
        print(f"  {hits}/{N} rays hit the mesh")

        camera_data[tag] = dict(
            origin=origin,
            C=C,
            origins=origins_arr,
            directions=directions,
            intersections=intersections,
            color=color,
        )

    # ------------------------------------------------------------------
    # 3. Stream to rerun
    # ------------------------------------------------------------------
    print("\nStarting rerun viewer...")
    rr.init("rhino_tracking", spawn=True)

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="Rhino 3D Tracking", origin="world"),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    # Static: colored point cloud
    print("Logging point cloud...")
    rr.log("world/points",
           rr.Points3D(positions=pc_verts, colors=pc_colors, radii=0.005),
           static=True)

    # Project point cloud colors onto mesh vertices via nearest-neighbor lookup
    print("Projecting point cloud colors onto mesh...")
    from scipy.spatial import KDTree
    kd = KDTree(pc_verts)
    _, idx = kd.query(intersector.mesh.vertices, workers=-1)
    mesh_colors = pc_colors[idx]   # (N_verts, 3) uint8

    # Static: colored mesh
    print("Logging mesh...")
    rr.log("world/mesh",
           rr.Mesh3D(
               vertex_positions=intersector.mesh.vertices,
               triangle_indices=intersector.mesh.faces,
               vertex_colors=mesh_colors,
           ),
           static=True)

    # Static: camera position markers
    for tag, d in camera_data.items():
        rr.log(f"world/cameras/{tag}",
               rr.Points3D(positions=[d['C']], colors=[d['color']], radii=0.2),
               static=True)

    # Static: rhino model geometry (centered at origin, feet at Y=0)
    print("Loading rhino model...")
    rhino_mesh = trimesh.load(str(RHINO_MODEL), force='mesh')

    # COLMAP world Y points DOWN; the model has Y pointing UP — flip it.
    rhino_verts = rhino_mesh.vertices.copy()
    rhino_verts[:, 1] *= -1          # flip Y
    # Flipping Y reverses face winding (would make normals inside-out), so flip faces too
    rhino_faces = rhino_mesh.faces[:, ::-1].copy()
    # After Y-flip: feet are at max Y (ground in world space)
    # Translate so feet sit at Y=0, body goes up in -Y direction
    rhino_verts[:, 1] -= rhino_verts[:, 1].max()
    # Apply scale
    rhino_verts *= args.rhino_scale
    # Recompute normals on the corrected mesh for shading
    corrected = trimesh.Trimesh(vertices=rhino_verts, faces=rhino_faces, process=False)
    rr.log(
        "world/rhino/model",
        rr.Mesh3D(
            vertex_positions=rhino_verts,
            triangle_indices=rhino_faces,
            vertex_normals=corrected.vertex_normals,
        ),
        static=True,
    )
    print(f"  Rhino model loaded: {len(rhino_mesh.vertices):,} vertices")

    # Per-frame: ray + ground dot
    max_frames = max(len(d['intersections']) for d in camera_data.values())
    print(f"Logging {max_frames} frames to rerun timeline...")

    for frame_idx in range(max_frames):
        rr.set_time("frame", sequence=frame_idx * args.every)

        for tag, d in camera_data.items():
            if frame_idx >= len(d['intersections']):
                continue

            origin    = d['origins'][frame_idx]
            direction = d['directions'][frame_idx]
            pt        = d['intersections'][frame_idx]
            color     = d['color']

            if not np.any(np.isnan(pt)):
                ray_end = pt
                # Move rhino model to intersection point
                rr.log("world/rhino", rr.Transform3D(translation=pt))
            else:
                ray_end = origin + direction * 15.0

            rr.log(f"world/{tag}/ray",
                   rr.LineStrips3D(strips=[[origin.tolist(), ray_end.tolist()]],
                                   colors=[color], radii=0.02))

    print("\nAll data logged. Use the timeline scrubber at the bottom to play.")


if __name__ == '__main__':
    main()
