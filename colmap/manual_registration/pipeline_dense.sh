#!/bin/bash
set -e

# Dense reconstruction from existing sparse model
# Usage: ./pipeline_dense.sh <sparse_model_path> <image_path> <dense_output_dir>

# Container setup
LOCAL_CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$LOCAL_CONTAINER"
CONTAINER="$LOCAL_CONTAINER"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

SPARSE_PATH="${1:?Usage: $0 <sparse_model_path> <image_path> <dense_output_dir>}"
IMAGE_PATH="${2:?Usage: $0 <sparse_model_path> <image_path> <dense_output_dir>}"
DENSE_DIR="${3:-./dense}"

mkdir -p "$DENSE_DIR"

echo "============================================================================"
echo "Dense Reconstruction Pipeline (max quality)"
echo "  sparse model : $SPARSE_PATH"
echo "  images       : $IMAGE_PATH"
echo "  output       : $DENSE_DIR"
echo "============================================================================"

# Step 1 — undistort images at full resolution, use more source images per view
echo ""
echo "=== [1/4] Image undistortion ==="
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$SPARSE_PATH" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size -1 \
    --num_patch_match_src_images 40

# Step 2 — patch match stereo (GPU)
#   window_radius=5       patch size (5 = fine detail, larger = smoother)
#   window_step=1         sample every pixel (max density)
#   num_samples=20        random hypotheses per pixel (up from 15)
#   num_iterations=8      propagation iterations (up from 5)
#   geom_consistency=1    cross-view depth consistency check (on by default)
#   filter_min_num_consistent=2  keep points seen in >=2 views (completeness)
#   cache_size=64         raise from 32 to use available RAM
echo ""
echo "=== [2/4] Patch match stereo (MVS) ==="
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

# Step 3 — fuse depth maps into dense point cloud
#   min_num_pixels=3      keep points visible in >=3 views (down from 5, more complete)
#   max_reproj_error=1    tighter reprojection filter (down from 2, more accurate)
#   max_depth_error=0.01  tight depth consistency
#   max_normal_error=10   normal consistency threshold
#   check_num_images=50   check consistency across up to 50 images
#   cache_size=64         raise from 32
echo ""
echo "=== [3/4] Stereo fusion ==="
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

# Step 4 — Poisson mesh
#   depth=14              octree depth for mesh resolution (up from 13)
#   trim=7                trim loose surfaces less aggressively (down from 10)
#   num_threads=-1        use all available cores
echo ""
echo "=== [4/4] Poisson meshing ==="
colmap poisson_mesher \
    --input_path "$DENSE_DIR/fused.ply" \
    --output_path "$DENSE_DIR/meshed-poisson.ply" \
    --PoissonMeshing.depth 14 \
    --PoissonMeshing.point_weight 1 \
    --PoissonMeshing.trim 7 \
    --PoissonMeshing.num_threads -1

echo ""
echo "============================================================================"
echo "Dense reconstruction complete!"
echo "  fused point cloud : $DENSE_DIR/fused.ply"
echo "  mesh              : $DENSE_DIR/meshed-poisson.ply"
echo "============================================================================"

rm -f "$LOCAL_CONTAINER"
