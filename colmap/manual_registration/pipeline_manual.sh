#!/bin/bash
set -e

# Pipeline: Default COLMAP for handheld, then manual PnP registration for stationary
#
# Output structure:
#   01_handheld/              <- handheld-only reconstruction
#   02_with_stationary/       <- best model + stationary cameras via manual PnP
#     sparse/registered/      <- after PnP registration (text format)
#     sparse/registered_bin/  <- converted to binary
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
echo "Pipeline: Manual Registration"
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
# STAGE 2: Register stationary cameras via manual correspondences
# ============================================================================

echo ""
echo "====== STAGE 2: Adding stationary cameras (Manual PnP) ======"
rm -rf "$STAGE2_DIR"
mkdir -p "$STAGE2_DIR/sparse/registered"
mkdir -p "$STAGE2_DIR/sparse/optimized"

echo ""
echo "[2.1] Running manual PnP registration..."
python3 "$SCRIPT_DIR/register_manual.py" \
    --model_path "$BEST_MODEL" \
    --correspondences "$SCRIPT_DIR/correspondences.json" \
    --output_path "$STAGE2_DIR/sparse/registered" \
    --stationary_dir "$STATIONARY_DIR"

echo ""
echo "[2.2] Converting text model to binary for bundle adjustment..."
mkdir -p "$STAGE2_DIR/sparse/registered_bin"
colmap model_converter \
    --input_path "$STAGE2_DIR/sparse/registered" \
    --output_path "$STAGE2_DIR/sparse/registered_bin" \
    --output_type BIN

echo ""
echo "[2.3] Bundle adjustment (refining all poses)..."
colmap bundle_adjuster \
    --input_path "$STAGE2_DIR/sparse/registered_bin" \
    --output_path "$STAGE2_DIR/sparse/optimized" \
    --BundleAdjustment.refine_focal_length 1 \
    --BundleAdjustment.refine_extra_params 1

# ============================================================================
# DONE
# ============================================================================
echo ""
echo "============================================================================"
echo "Pipeline complete!"
echo "============================================================================"
echo ""
echo "=== Results (after optimization) ==="
colmap model_analyzer --path "$STAGE2_DIR/sparse/optimized"

echo ""
echo "Stage 1 output (handheld only): $STAGE1_DIR/sparse/"
echo "Stage 2 registered (text): $STAGE2_DIR/sparse/registered/"
echo "Stage 2 optimized: $STAGE2_DIR/sparse/optimized/"

rm -f "$LOCAL_CONTAINER"
