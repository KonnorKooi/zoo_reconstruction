#!/bin/bash
set -e

# Dense reconstruction from a COLMAP sparse model + undistorted images.
# Designed to run as an HTCondor job on the CSCI cluster.
# No arguments needed — all paths are absolute and hardcoded below.

# ── Absolute paths ────────────────────────────────────────────────────────────
SANDBOX="$(pwd)"   # HTCondor scratch SSD: /scratch_ssd/condor/execute/dir_XXXX

CONTAINER_SRC="/cluster/research-groups/wehrwein/zoo/containers/colmap.sif"
LOCAL_CONTAINER="/tmp/colmap_$$.sif"

# Sparse model on NFS — bin files fetched once at job start (small sequential read)
NFS_SPARSE="/cluster/research-groups/wehrwein/zoo/konnor/colmap/zoo_reconstruction/colmap/manual_registration/sparse/1"

# Sandbox paths — images are transferred here by HTCondor, bin files copied below
SPARSE_DIR="$SANDBOX/1"   # HTCondor transfers sparse/1 using basename → sandbox/1/
DENSE_DIR="$SANDBOX/dense"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================================"
echo "Dense Reconstruction Pipeline"
echo "  sandbox   : $SANDBOX"
echo "  sparse    : $SPARSE_DIR"
echo "  nfs model : $NFS_SPARSE"
echo "  output    : $DENSE_DIR"
echo "============================================================================"

# Copy container to local SSD (faster than reading from NFS during exec)
echo ""
echo "=== [0/4] Staging container ==="
cp "$CONTAINER_SRC" "$LOCAL_CONTAINER"
colmap() { apptainer exec --nv "$LOCAL_CONTAINER" colmap "$@"; }

# HTCondor transfers images/ subdir but skips top-level .bin files (cluster quirk).
# Copy the 3 small model files from NFS once — one-time sequential read, negligible load.
echo ""
echo "=== [1/4] Fetching sparse model from NFS ==="
mkdir -p "$SPARSE_DIR"
cp "$NFS_SPARSE/cameras.bin"  "$SPARSE_DIR/cameras.bin"
cp "$NFS_SPARSE/images.bin"   "$SPARSE_DIR/images.bin"
cp "$NFS_SPARSE/points3D.bin" "$SPARSE_DIR/points3D.bin"

# Sanity checks
[ -f "$SPARSE_DIR/cameras.bin" ] || { echo "ERROR: cameras.bin missing after NFS copy"; exit 1; }
[ -d "$SPARSE_DIR/images"      ] || { echo "ERROR: images/ not found — HTCondor transfer failed"; exit 1; }
echo "    sparse model ready."

# Wipe any previous dense output and start clean
rm -rf "$DENSE_DIR"
mkdir -p "$DENSE_DIR"

# Step 2 — undistort images and generate stereo workspace configs
#   cameras are already pinhole so this mainly generates patch-match.cfg
#   num_patch_match_src_images=40  more source views → better depth estimates
echo ""
echo "=== [2/4] Image undistortion ==="
colmap image_undistorter \
    --image_path    "$SPARSE_DIR/images" \
    --input_path    "$SPARSE_DIR" \
    --output_path   "$DENSE_DIR" \
    --output_type   COLMAP \
    --max_image_size -1 \
    --num_patch_match_src_images 40

# Step 3 — patch match stereo (GPU)
echo ""
echo "=== [3/4] Patch match stereo (MVS) ==="
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size -1 \
    --PatchMatchStereo.window_radius 5 \
    --PatchMatchStereo.window_step 1 \
    --PatchMatchStereo.num_samples 20 \
    --PatchMatchStereo.num_iterations 8 \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.geom_consistency_regularizer 0.3 \
    --PatchMatchStereo.geom_consistency_max_cost 3 \
    --PatchMatchStereo.filter 1 \
    --PatchMatchStereo.filter_min_ncc 0.1 \
    --PatchMatchStereo.filter_min_triangulation_angle 3 \
    --PatchMatchStereo.filter_min_num_consistent 2 \
    --PatchMatchStereo.cache_size 64

# Step 4 — fuse depth maps then Poisson mesh
echo ""
echo "=== [4/4] Stereo fusion + Poisson mesh ==="
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply" \
    --StereoFusion.max_image_size -1 \
    --StereoFusion.min_num_pixels 3 \
    --StereoFusion.max_num_pixels 10000 \
    --StereoFusion.max_traversal_depth 100 \
    --StereoFusion.max_reproj_error 1 \
    --StereoFusion.max_depth_error 0.01 \
    --StereoFusion.max_normal_error 10 \
    --StereoFusion.check_num_images 50 \
    --StereoFusion.cache_size 64

colmap poisson_mesher \
    --input_path  "$DENSE_DIR/fused.ply" \
    --output_path "$DENSE_DIR/meshed-poisson.ply" \
    --PoissonMeshing.depth 14 \
    --PoissonMeshing.point_weight 1 \
    --PoissonMeshing.trim 7 \
    --PoissonMeshing.num_threads -1

echo ""
echo "============================================================================"
echo "Done!"
echo "  point cloud : $DENSE_DIR/fused.ply"
echo "  mesh        : $DENSE_DIR/meshed-poisson.ply"
echo "============================================================================"

rm -f "$LOCAL_CONTAINER"
