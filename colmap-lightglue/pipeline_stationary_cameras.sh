#!/bin/bash
#
# Pipeline for reconstructing scenes with stationary + handheld cameras
# Uses COLMAP (SIFT) for handheld, LightGlue for stationary-to-handheld matching
#
# Requirements:
#   - COLMAP installed in WSL: sudo apt install colmap
#   - Python with: pip install lightglue torch torchvision
#
# Usage:
#   ./pipeline_stationary_cameras.sh /path/to/project
#
# Expected directory structure:
#   /path/to/project/
#     handheld/     <- handheld camera images
#     stationary/   <- stationary camera images (elevated/far views)
#

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Adjust these as needed
# ============================================================================

# container setup to use the colmap in the apptainer
CONTAINER="/cluster/research-groups/wehrwein/zoo/containers/colmap.sif"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

# Feature extraction settings
HANDHELD_MAX_FEATURES=5000        # Features per handheld image
STATIONARY_MAX_FEATURES=16384      # Features per stationary image

# LightGlue matching settings
LIGHTGLUE_MAX_KEYPOINT_DIST=10    # Max distance to match LightGlue point to SIFT keypoint
LIGHTGLUE_MIN_MATCHES=15          # Minimum matches to keep a pair
LIGHTGLUE_STATIONARY_MAX_KP=32768 # More keypoints for stationary cameras
LIGHTGLUE_STATIONARY_THRESHOLD=0.001  # Lower threshold = more features
LIGHTGLUE_MULTISCALE=true         # Enable multi-scale extraction for stationary

# COLMAP mapper settings (relaxed for difficult matching)
MAPPER_MIN_NUM_MATCHES=10
MAPPER_INIT_MIN_NUM_INLIERS=15
MAPPER_ABS_POSE_MIN_INLIERS=12
MAPPER_ABS_POSE_MIN_INLIER_RATIO=0.05
MAPPER_TRI_MIN_ANGLE=0.5

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/project"
    echo ""
    echo "Expected structure:"
    echo "  /path/to/project/"
    echo "    handheld/     <- handheld camera images"
    echo "    stationary/   <- stationary camera images"
    exit 1
fi

PROJECT_DIR="."
HANDHELD_DIR="./handheld"
STATIONARY_DIR="./stationary"
DATABASE_PATH="./database.db"
SPARSE_DIR="./sparse"

# Verify directories exist
if [ ! -d "$HANDHELD_DIR" ]; then
    echo "ERROR: Handheld directory not found: $HANDHELD_DIR"
    exit 1
fi

if [ ! -d "$STATIONARY_DIR" ]; then
    echo "ERROR: Stationary directory not found: $STATIONARY_DIR"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "============================================================================"
echo "Pipeline: Stationary + Handheld Camera Reconstruction"
echo "============================================================================"
echo "Project:    $PROJECT_DIR"
echo "Handheld:   $HANDHELD_DIR"
echo "Stationary: $STATIONARY_DIR"
echo "Database:   $DATABASE_PATH"
echo "Output:     $SPARSE_DIR"
echo ""

# ============================================================================
# STEP 1: Clean up previous run
# ============================================================================

echo "[Step 1/7] Cleaning up previous run..."
rm -f "$DATABASE_PATH"
rm -rf "$SPARSE_DIR"
mkdir -p "$SPARSE_DIR"

# ============================================================================
# STEP 2: Extract SIFT features for handheld images
# ============================================================================

echo ""
echo "[Step 2/7] Extracting SIFT features for handheld images..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$HANDHELD_DIR" \
    --SiftExtraction.max_num_features $HANDHELD_MAX_FEATURES \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

# ============================================================================
# STEP 3: Extract SIFT features for stationary images (with upscaling)
# ============================================================================

echo ""
echo "[Step 3/7] Extracting SIFT features for stationary images..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$STATIONARY_DIR" \
    --SiftExtraction.max_num_features $STATIONARY_MAX_FEATURES \
    --SiftExtraction.first_octave -1 \
    --SiftExtraction.num_octaves 6 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1 \
    --SiftExtraction.dsp_min_scale 0.1 \
    --SiftExtraction.dsp_max_scale 4.0 \
    --ImageReader.single_camera 1 \
    --ImageReader.default_focal_length_factor 1.2

# ============================================================================
# STEP 4: Match handheld images with COLMAP
# ============================================================================

echo ""
echo "[Step 4/7] Matching handheld images with COLMAP..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

# ============================================================================
# STEP 5: Run LightGlue matching for stationary -> handheld
# ============================================================================

echo ""
echo "[Step 5/7] Running LightGlue matching for stationary -> handheld..."
MULTISCALE_FLAG=""
if [ "$LIGHTGLUE_MULTISCALE" = true ]; then
    MULTISCALE_FLAG="--multiscale --scales 0.5,1.0,2.0"
fi
apptainer exec --nv "$CONTAINER" python3 "$SCRIPT_DIR/lightglue_match.py" \
    --database_path "$DATABASE_PATH" \
    --stationary_dir "$STATIONARY_DIR" \
    --handheld_dir "$HANDHELD_DIR" \
    --max_keypoint_dist $LIGHTGLUE_MAX_KEYPOINT_DIST \
    --min_matches $LIGHTGLUE_MIN_MATCHES \
    --stationary_max_keypoints $LIGHTGLUE_STATIONARY_MAX_KP \
    --stationary_detection_threshold $LIGHTGLUE_STATIONARY_THRESHOLD \
    $MULTISCALE_FLAG

# ============================================================================
# STEP 6: Initial reconstruction (handheld images)
# ============================================================================

echo ""
echo "[Step 6/7] Running initial reconstruction..."
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "." \
    --output_path "$SPARSE_DIR" \
    --Mapper.min_num_matches $MAPPER_MIN_NUM_MATCHES \
    --Mapper.init_min_num_inliers $MAPPER_INIT_MIN_NUM_INLIERS \
    --Mapper.tri_min_angle $MAPPER_TRI_MIN_ANGLE

# Find the largest reconstruction
BEST_MODEL=$(ls -d "$SPARSE_DIR"/*/ 2>/dev/null | head -1)
if [ -z "$BEST_MODEL" ]; then
    echo "ERROR: No reconstruction created"
    exit 1
fi
echo "Best model: $BEST_MODEL"

# ============================================================================
# STEP 7: Register stationary cameras with relaxed constraints
# ============================================================================

echo ""
echo "[Step 7/7] Registering stationary cameras with relaxed constraints..."
colmap image_registrator \
    --database_path "$DATABASE_PATH" \
    --input_path "$BEST_MODEL" \
    --output_path "$SPARSE_DIR/final" \
    --Mapper.abs_pose_min_num_inliers $MAPPER_ABS_POSE_MIN_INLIERS \
    --Mapper.abs_pose_min_inlier_ratio $MAPPER_ABS_POSE_MIN_INLIER_RATIO

# Add this after Step 7
# Triangulate additional points using the registered stationary camera poses
# This fills in gaps in the 3D point cloud by computing 3D points from matched features
colmap point_triangulator \
    --database_path "$DATABASE_PATH" \
    --image_path "$PROJECT_DIR" \
    --input_path "$SPARSE_DIR/final" \
    --output_path "$SPARSE_DIR/final"

# Refine the reconstruction through bundle adjustment
# Optimizes camera poses and 3D point positions to minimize reprojection error
# Also refines focal length and lens distortion parameters for better accuracy
colmap bundle_adjuster \
    --input_path "$SPARSE_DIR/final" \
    --output_path "$SPARSE_DIR/final_optimized" \
    --BundleAdjustment.refine_focal_length 1 \
    --BundleAdjustment.refine_extra_params 1 \


# ============================================================================
# DONE
# ============================================================================

echo ""
echo "============================================================================"
echo "Pipeline complete!"
echo "============================================================================"
echo ""
colmap model_analyzer --path "$SPARSE_DIR/final_optimized"

echo ""
echo "Output: $SPARSE_DIR/final"
echo ""
echo "To view in COLMAP GUI:"
echo "  colmap gui"
echo "  File -> Import model -> $SPARSE_DIR/final"
