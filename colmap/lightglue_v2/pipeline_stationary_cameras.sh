#!/bin/bash
set -e

# Pipeline v2: Default COLMAP for handheld, then add stationary cameras via LightGlue
#
# v2 improvements over v1:
#   - Increased max_keypoint_dist (10 -> 25) for more SIFT-SuperPoint bridging
#   - More scales (7 instead of 3) for better scale coverage
#   - Lower detection threshold (0.001 -> 0.0005) for more SuperPoint features
#   - Added point triangulation + bundle adjustment after registration
#
# Output structure:
#   01_handheld/          <- handheld-only reconstruction (multiple models)
#     database.db
#     sparse/0/, sparse/1/, ...
#   02_with_stationary/   <- best handheld model + stationary cameras added
#     database.db
#     sparse/final/
#     sparse/final_optimized/

# Container setup
LOCAL_CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$LOCAL_CONTAINER"
CONTAINER="$LOCAL_CONTAINER"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

# LightGlue settings (v2: relaxed for more matches)
LIGHTGLUE_MAX_KEYPOINT_DIST=25
LIGHTGLUE_MIN_MATCHES=15
LIGHTGLUE_STATIONARY_MAX_KP=32768
LIGHTGLUE_STATIONARY_THRESHOLD=0.0005
LIGHTGLUE_MULTISCALE=true
LIGHTGLUE_SCALES="0.25,0.5,0.75,1.0,1.5,2.0,3.0"

# Stationary registration settings
MAPPER_ABS_POSE_MIN_INLIERS=12
MAPPER_ABS_POSE_MIN_INLIER_RATIO=0.05

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/project"
    exit 1
fi

HANDHELD_DIR="./handheld"
STATIONARY_DIR="./stationary"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

STAGE1_DIR="./01_handheld"
STAGE2_DIR="./02_with_stationary"

echo "============================================================================"
echo "Pipeline v2: Stationary + Handheld Camera Reconstruction"
echo "============================================================================"
echo "LightGlue max_keypoint_dist: $LIGHTGLUE_MAX_KEYPOINT_DIST"
echo "LightGlue detection_threshold: $LIGHTGLUE_STATIONARY_THRESHOLD"
echo "LightGlue scales: $LIGHTGLUE_SCALES"
echo ""

# ============================================================================
# STAGE 1: Handheld-only reconstruction
# ============================================================================

echo ""
echo "====== STAGE 1: Handheld-only reconstruction ======"
rm -rf "$STAGE1_DIR"
mkdir -p "$STAGE1_DIR/sparse"

echo ""
echo "[1.1] Extracting features for handheld images..."
colmap feature_extractor \
    --database_path "$STAGE1_DIR/database.db" \
    --image_path "$HANDHELD_DIR"

echo ""
echo "[1.2] Matching handheld images..."
colmap exhaustive_matcher \
    --database_path "$STAGE1_DIR/database.db"

echo ""
echo "[1.3] Reconstructing handheld images..."
colmap mapper \
    --database_path "$STAGE1_DIR/database.db" \
    --image_path "$HANDHELD_DIR" \
    --output_path "$STAGE1_DIR/sparse"

# Report all models
echo ""
echo "=== Stage 1 Results ==="
for d in "$STAGE1_DIR/sparse"/*/; do
    echo "Model: $d"
    colmap model_analyzer --path "$d" 2>&1
    echo ""
done

# Find the largest model
BEST_MODEL=""
BEST_COUNT=0
for d in "$STAGE1_DIR/sparse"/*/; do
    COUNT=$(colmap model_analyzer --path "$d" 2>&1 | grep "Registered images" | awk '{print $NF}' || echo 0)
    if [ "$COUNT" -gt "$BEST_COUNT" ]; then
        BEST_COUNT=$COUNT
        BEST_MODEL=$d
    fi
done

if [ -z "$BEST_MODEL" ]; then
    echo "ERROR: No reconstruction created"
    exit 1
fi
echo "Best model: $BEST_MODEL ($BEST_COUNT images)"

# ============================================================================
# STAGE 2: Add stationary cameras to best handheld model
# ============================================================================

echo ""
echo "====== STAGE 2: Adding stationary cameras ======"
rm -rf "$STAGE2_DIR"
mkdir -p "$STAGE2_DIR/sparse/final"
mkdir -p "$STAGE2_DIR/sparse/final_optimized"

# Copy the handheld database as starting point
cp "$STAGE1_DIR/database.db" "$STAGE2_DIR/database.db"

echo ""
echo "[2.1] Extracting features for stationary images..."
colmap feature_extractor \
    --database_path "$STAGE2_DIR/database.db" \
    --image_path "$STATIONARY_DIR" \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.first_octave -1 \
    --SiftExtraction.num_octaves 6 \
    --ImageReader.single_camera 1 \
    --ImageReader.default_focal_length_factor 1.2

echo ""
echo "[2.2] Running LightGlue matching for stationary -> handheld..."
MULTISCALE_FLAG=""
if [ "$LIGHTGLUE_MULTISCALE" = true ]; then
    MULTISCALE_FLAG="--multiscale --scales $LIGHTGLUE_SCALES"
fi
apptainer exec --nv "$CONTAINER" python3 "$SCRIPT_DIR/lightglue_match.py" \
    --database_path "$STAGE2_DIR/database.db" \
    --stationary_dir "$STATIONARY_DIR" \
    --handheld_dir "$HANDHELD_DIR" \
    --max_keypoint_dist $LIGHTGLUE_MAX_KEYPOINT_DIST \
    --min_matches $LIGHTGLUE_MIN_MATCHES \
    --stationary_max_keypoints $LIGHTGLUE_STATIONARY_MAX_KP \
    --stationary_detection_threshold $LIGHTGLUE_STATIONARY_THRESHOLD \
    $MULTISCALE_FLAG

echo ""
echo "[2.3] Registering stationary cameras into handheld model..."
colmap image_registrator \
    --database_path "$STAGE2_DIR/database.db" \
    --input_path "$BEST_MODEL" \
    --output_path "$STAGE2_DIR/sparse/final" \
    --Mapper.abs_pose_min_num_inliers $MAPPER_ABS_POSE_MIN_INLIERS \
    --Mapper.abs_pose_min_inlier_ratio $MAPPER_ABS_POSE_MIN_INLIER_RATIO

echo ""
echo "[2.4] Triangulating additional points..."
# Create a combined image directory so triangulator can find all images
mkdir -p ./all_images
ln -sf "$(pwd)/$HANDHELD_DIR"/* ./all_images/
ln -sf "$(pwd)/$STATIONARY_DIR"/* ./all_images/
colmap point_triangulator \
    --database_path "$STAGE2_DIR/database.db" \
    --image_path ./all_images \
    --input_path "$STAGE2_DIR/sparse/final" \
    --output_path "$STAGE2_DIR/sparse/final"
rm -rf ./all_images

echo ""
echo "[2.5] Bundle adjustment (refining all poses)..."
colmap bundle_adjuster \
    --input_path "$STAGE2_DIR/sparse/final" \
    --output_path "$STAGE2_DIR/sparse/final_optimized" \
    --BundleAdjustment.refine_focal_length 1 \
    --BundleAdjustment.refine_extra_params 1

# ============================================================================
# DONE
# ============================================================================
echo ""
echo "============================================================================"
echo "Pipeline v2 complete!"
echo "============================================================================"
echo ""
echo "=== Stage 2 Results (after optimization) ==="
colmap model_analyzer --path "$STAGE2_DIR/sparse/final_optimized"

echo ""
echo "Stage 1 output (handheld only): $STAGE1_DIR/sparse/"
echo "Stage 2 output (before optimization): $STAGE2_DIR/sparse/final/"
echo "Stage 2 output (after optimization): $STAGE2_DIR/sparse/final_optimized/"

rm -f "$LOCAL_CONTAINER"
