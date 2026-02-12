#!/bin/bash
set -e

# Pipeline v2: Default COLMAP for handheld, then direct PnP registration for stationary
#
# v2 improvements:
#   - Direct PnP pose estimation instead of image_registrator
#   - Bridges SuperPoint matches to SIFT keypoints with known 3D points
#   - More scales and lower detection threshold for more features
#   - Bundle adjustment after registration to refine all poses
#
# Output structure:
#   01_handheld/              <- handheld-only reconstruction
#   02_with_stationary/       <- best model + stationary cameras via PnP
#     sparse/registered/      <- after PnP registration
#     sparse/optimized/       <- after bundle adjustment

# Container setup
LOCAL_CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$LOCAL_CONTAINER"
CONTAINER="$LOCAL_CONTAINER"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

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
echo "Pipeline v2: Stationary + Handheld Camera Reconstruction (Direct PnP)"
echo "============================================================================"

# ============================================================================
# STAGE 1: Handheld-only reconstruction (default COLMAP)
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
# STAGE 2: Register stationary cameras via direct PnP
# ============================================================================

echo ""
echo "====== STAGE 2: Adding stationary cameras (Direct PnP) ======"
rm -rf "$STAGE2_DIR"
mkdir -p "$STAGE2_DIR/sparse/registered"
mkdir -p "$STAGE2_DIR/sparse/optimized"

echo ""
echo "[2.1] Running direct PnP registration..."
apptainer exec --nv "$CONTAINER" python3 "$SCRIPT_DIR/register_stationary.py" \
    --model_path "$BEST_MODEL" \
    --database_path "$STAGE1_DIR/database.db" \
    --output_path "$STAGE2_DIR/sparse/registered" \
    --stationary_dir "$STATIONARY_DIR" \
    --handheld_dir "$HANDHELD_DIR" \
    --max_keypoint_dist 25 \
    --min_correspondences 12 \
    --pnp_reproj_threshold 8.0 \
    --stationary_max_keypoints 32768 \
    --stationary_detection_threshold 0.0005 \
    --multiscale \
    --scales "0.25,0.5,0.75,1.0,1.5,2.0,3.0"

echo ""
echo "[2.2] Bundle adjustment (refining all poses)..."
colmap bundle_adjuster \
    --input_path "$STAGE2_DIR/sparse/registered" \
    --output_path "$STAGE2_DIR/sparse/optimized" \
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
echo "=== Results (after optimization) ==="
colmap model_analyzer --path "$STAGE2_DIR/sparse/optimized"

echo ""
echo "Stage 1 output (handheld only): $STAGE1_DIR/sparse/"
echo "Stage 2 registered: $STAGE2_DIR/sparse/registered/"
echo "Stage 2 optimized: $STAGE2_DIR/sparse/optimized/"

rm -f "$LOCAL_CONTAINER"
