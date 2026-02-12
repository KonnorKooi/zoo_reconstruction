#!/bin/bash
set -e

# High-quality sparse + dense reconstruction — handheld images only
# Optimized for maximum point cloud density

CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$CONTAINER"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

SPARSE_DIR="./sparse"
DENSE_DIR="./dense"

rm -f database.db
rm -rf "$SPARSE_DIR" "$DENSE_DIR"
mkdir -p "$SPARSE_DIR"

# ============================================================================
# STAGE 1: High-quality sparse reconstruction
# ============================================================================

echo "============================================================================"
echo "STAGE 1: Sparse reconstruction (high quality)"
echo "============================================================================"

echo ""
echo "[1.1] Feature extraction (16k features, double resolution first octave)..."
colmap feature_extractor \
    --database_path database.db \
    --image_path ./handheld \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.first_octave -1 \
    --SiftExtraction.num_octaves 6 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

echo ""
echo "[1.2] Exhaustive matching..."
colmap exhaustive_matcher \
    --database_path database.db

echo ""
echo "[1.3] Mapper..."
colmap mapper \
    --database_path database.db \
    --image_path ./handheld \
    --output_path "$SPARSE_DIR"

# Find the largest model
echo ""
echo "=== Sparse Results ==="
BEST_MODEL=""
BEST_COUNT=0
for d in "$SPARSE_DIR"/*/; do
    COUNT=$(colmap model_analyzer --path "$d" 2>&1 | grep "Registered images" | awk '{print $NF}' || echo 0)
    echo "Model $d: $COUNT images"
    colmap model_analyzer --path "$d" 2>&1
    echo ""
    if [ "$COUNT" -gt "$BEST_COUNT" ]; then
        BEST_COUNT=$COUNT
        BEST_MODEL=$d
    fi
done

if [ -z "$BEST_MODEL" ]; then
    echo "ERROR: No reconstruction created"
    rm -f "$CONTAINER"
    exit 1
fi
echo "Best model: $BEST_MODEL ($BEST_COUNT images)"

# ============================================================================
# STAGE 2: Dense reconstruction
# ============================================================================

echo ""
echo "============================================================================"
echo "STAGE 2: Dense reconstruction"
echo "============================================================================"

echo ""
echo "[2.1] Undistorting images (full resolution)..."
colmap image_undistorter \
    --image_path ./handheld \
    --input_path "$BEST_MODEL" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size -1 \
    --num_patch_match_src_images 30

echo ""
echo "[2.2] Patch match stereo (high quality, 10 iterations)..."
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size -1 \
    --PatchMatchStereo.window_radius 7 \
    --PatchMatchStereo.num_iterations 10 \
    --PatchMatchStereo.num_samples 15 \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.filter_min_ncc 0.1 \
    --PatchMatchStereo.min_triangulation_angle 1 \
    --PatchMatchStereo.filter_min_num_consistent 2

echo ""
echo "[2.3] Stereo fusion (relaxed filtering for max density)..."
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply" \
    --StereoFusion.min_num_pixels 3 \
    --StereoFusion.max_reproj_error 2 \
    --StereoFusion.max_depth_error 0.02 \
    --StereoFusion.max_normal_error 10 \
    --StereoFusion.check_num_images 50

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "============================================================================"
echo "Done!"
echo "============================================================================"
echo ""
echo "Sparse reconstruction: $BEST_MODEL"
colmap model_analyzer --path "$BEST_MODEL" 2>&1
echo ""
echo "Dense point cloud: $DENSE_DIR/fused.ply"
if [ -f "$DENSE_DIR/fused.ply" ]; then
    echo "File size: $(du -h "$DENSE_DIR/fused.ply" | cut -f1)"
fi

rm -f "$CONTAINER"
