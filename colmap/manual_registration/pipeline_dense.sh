#!/bin/bash
set -e

# Dense reconstruction from a prepared COLMAP MVS workspace.
# Cleans up any previous run's intermediate files before starting.
#
# Usage: ./pipeline_dense.sh <workspace_dir> [dense_output_dir]
#
# <workspace_dir>   path to the COLMAP workspace produced by image_undistorter
#                   (must contain images/, sparse/, stereo/patch-match.cfg)
# [dense_output_dir] optional override for output; defaults to <workspace_dir>

# Container setup
LOCAL_CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$LOCAL_CONTAINER"
CONTAINER="$LOCAL_CONTAINER"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

WORKSPACE="${1:?Usage: $0 <workspace_dir>}"
DENSE_DIR="${2:-$WORKSPACE}"

echo "============================================================================"
echo "Dense Reconstruction Pipeline (max quality)"
echo "  workspace : $WORKSPACE"
echo "  output    : $DENSE_DIR"
echo "============================================================================"

# Guard: sparse/ subdir must have a valid camera model — if missing, the job
# was likely submitted before committing sparse/1/sparse/*.bin to git.
if [ ! -f "$WORKSPACE/sparse/cameras.bin" ]; then
    echo ""
    echo "ERROR: $WORKSPACE/sparse/cameras.bin not found."
    echo "Commit sparse/1/sparse/*.bin to git and re-push before submitting."
    exit 1
fi

# Wipe previous run's intermediates so we always start clean
echo ""
echo "=== [1/3] Cleaning up previous outputs ==="
# Ensure stereo subdirs exist (HTCondor doesn't transfer empty directories)
mkdir -p "$WORKSPACE/stereo/depth_maps" \
         "$WORKSPACE/stereo/normal_maps" \
         "$WORKSPACE/stereo/consistency_graphs"
rm -f  "$WORKSPACE/stereo/depth_maps"/*.bin \
       "$WORKSPACE/stereo/depth_maps"/*.jpg \
       "$WORKSPACE/stereo/depth_maps"/*.png
rm -f  "$WORKSPACE/stereo/normal_maps"/*.bin \
       "$WORKSPACE/stereo/normal_maps"/*.jpg \
       "$WORKSPACE/stereo/normal_maps"/*.png
rm -f  "$WORKSPACE/stereo/consistency_graphs"/*.bin
rm -f  "$DENSE_DIR/fused.ply"
rm -f  "$DENSE_DIR/meshed-poisson.ply"
echo "    done."

# Step 2 — patch match stereo (GPU)
#   window_radius=5       patch size (5 = fine detail, larger = smoother)
#   window_step=1         sample every pixel (max density)
#   num_samples=20        random hypotheses per pixel
#   num_iterations=8      propagation iterations
#   geom_consistency=1    cross-view depth consistency check
#   filter_min_num_consistent=2  keep points seen in >=2 views
#   cache_size=64         raise from 32 to use available RAM
echo ""
echo "=== [2/3] Patch match stereo (MVS) ==="
colmap patch_match_stereo \
    --workspace_path "$WORKSPACE" \
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
#   min_num_pixels=3      keep points visible in >=3 views (more complete)
#   max_reproj_error=1    tight reprojection filter (more accurate)
#   max_depth_error=0.01  tight depth consistency
#   max_normal_error=10   normal consistency threshold
#   check_num_images=50   check consistency across up to 50 images
#   cache_size=64         raise from 32
echo ""
echo "=== [3/3] Stereo fusion + Poisson mesh ==="
colmap stereo_fusion \
    --workspace_path "$WORKSPACE" \
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
#   depth=14              octree depth for mesh resolution
#   trim=7                trim loose surfaces
#   num_threads=-1        use all available cores
echo ""
echo "=== Poisson meshing ==="
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
