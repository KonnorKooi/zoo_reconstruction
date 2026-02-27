#!/bin/bash
set -e

# Dense reconstruction from a COLMAP sparse model + undistorted images.
# Usage: ./pipeline_dense.sh <sparse_dir> [dense_output_dir]
#
# <sparse_dir>       directory containing cameras.bin, images.bin,
#                    points3D.bin, and an images/ subfolder
# [dense_output_dir] defaults to ./dense

# Container setup
LOCAL_CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$LOCAL_CONTAINER"
CONTAINER="$LOCAL_CONTAINER"
colmap() { apptainer exec --nv "$CONTAINER" colmap "$@"; }

SPARSE_DIR="${1:?Usage: $0 <sparse_dir> [dense_output_dir]}"
DENSE_DIR="${2:-./dense}"

echo "============================================================================"
echo "Dense Reconstruction Pipeline"
echo "  sparse : $SPARSE_DIR"
echo "  output : $DENSE_DIR"
echo "============================================================================"

# Sanity check
if [ ! -f "$SPARSE_DIR/cameras.bin" ]; then
    echo "ERROR: $SPARSE_DIR/cameras.bin not found."
    exit 1
fi
if [ ! -d "$SPARSE_DIR/images" ]; then
    echo "ERROR: $SPARSE_DIR/images/ not found."
    exit 1
fi

# Start clean — wipe any previous dense output
echo ""
echo "=== [1/4] Preparing workspace ==="
rm -rf "$DENSE_DIR"
mkdir -p "$DENSE_DIR"

# Step 2 — undistort images and build stereo workspace
#   cameras are already pinhole so this is mainly generating patch-match.cfg
#   num_patch_match_src_images=40  more source views -> better depth estimates
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

# Step 4 — fuse depth maps + mesh
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
