#!/bin/bash
set -e

# Dense reconstruction from a COLMAP sparse model + undistorted images.
# Designed to run as an HTCondor job on the CSCI cluster.
# No arguments needed — all paths are absolute and hardcoded below.

# ── Paths ──────────────────────────────────────────────────────────────────────
SANDBOX="$(pwd)"   # HTCondor scratch SSD: /scratch_ssd/condor/execute/dir_XXXX

# HTCondor transfers all inputs to sandbox root (basename destination):
#   colmap.sif, cameras.bin, images.bin, points3D.bin, images/
LOCAL_CONTAINER="$SANDBOX/colmap.sif"

# NFS paths — fallback only if cluster .bin-skip quirk prevents file transfer
NFS_CONTAINER="/cluster/research-groups/wehrwein/zoo/containers/colmap.sif"
NFS_SPARSE="/cluster/research-groups/wehrwein/zoo/konnor/colmap/zoo_reconstruction/colmap/manual_registration/sparse/1"

# SPARSE_DIR = sandbox root: bin files land here, images/ subdir lands here too
SPARSE_DIR="$SANDBOX"
DENSE_DIR="$SANDBOX/dense"
# ──────────────────────────────────────────────────────────────────────────────

echo "============================================================================"
echo "Dense Reconstruction Pipeline"
echo "  sandbox   : $SANDBOX"
echo "  sparse    : $SPARSE_DIR"
echo "  output    : $DENSE_DIR"
echo "============================================================================"

# ── [0/3] Container ────────────────────────────────────────────────────────────
echo ""
echo "=== [0/3] Verifying container ==="
CLEANUP_CONTAINER=0
if [ ! -f "$LOCAL_CONTAINER" ]; then
    echo "    colmap.sif not in sandbox — NFS fallback (transfer may have failed)"
    LOCAL_CONTAINER="/tmp/colmap_$$.sif"
    cp "$NFS_CONTAINER" "$LOCAL_CONTAINER"
    CLEANUP_CONTAINER=1
fi
colmap() { apptainer exec --nv "$LOCAL_CONTAINER" colmap "$@"; }
echo "    container: $LOCAL_CONTAINER"

# ── Verify sparse model on SSD ─────────────────────────────────────────────────
echo ""
echo "=== Verifying sparse model ==="
# Bin files should be in sandbox from HTCondor transfer; fall back to NFS if cluster
# .bin-skip quirk still applies to explicit file listings.
for f in cameras.bin images.bin points3D.bin; do
    if [ ! -f "$SPARSE_DIR/$f" ]; then
        echo "    $f not in sandbox — NFS fallback"
        cp "$NFS_SPARSE/$f" "$SPARSE_DIR/$f"
    fi
done

[ -f "$SPARSE_DIR/cameras.bin"  ] || { echo "ERROR: cameras.bin missing"; exit 1; }
[ -f "$SPARSE_DIR/images.bin"   ] || { echo "ERROR: images.bin missing"; exit 1; }
[ -f "$SPARSE_DIR/points3D.bin" ] || { echo "ERROR: points3D.bin missing"; exit 1; }
[ -d "$SPARSE_DIR/images"       ] || { echo "ERROR: images/ not found — HTCondor transfer failed"; exit 1; }
echo "    all inputs on local SSD."

# Wipe any previous dense output and start clean
rm -rf "$DENSE_DIR"
mkdir -p "$DENSE_DIR"

# ── [1/3] Image undistortion ───────────────────────────────────────────────────
# cameras are already pinhole so this mainly generates patch-match.cfg
echo ""
echo "=== [1/3] Image undistortion ==="
colmap image_undistorter \
    --image_path  "$SPARSE_DIR/images" \
    --input_path  "$SPARSE_DIR" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP

# ── [2/3] Patch match stereo (GPU) ────────────────────────────────────────────
echo ""
echo "=== [2/3] Patch match stereo (MVS) ==="
colmap patch_match_stereo \
    --workspace_path   "$DENSE_DIR" \
    --workspace_format COLMAP

# ── [3/3] Stereo fusion + Poisson mesh ────────────────────────────────────────
echo ""
echo "=== [3/3] Stereo fusion + Poisson mesh ==="
colmap stereo_fusion \
    --workspace_path   "$DENSE_DIR" \
    --workspace_format COLMAP \
    --output_path      "$DENSE_DIR/fused.ply"

colmap poisson_mesher \
    --input_path  "$DENSE_DIR/fused.ply" \
    --output_path "$DENSE_DIR/meshed-poisson.ply"

echo ""
echo "============================================================================"
echo "Done!"
echo "  point cloud : $DENSE_DIR/fused.ply"
echo "  mesh        : $DENSE_DIR/meshed-poisson.ply"
echo "============================================================================"

[ "$CLEANUP_CONTAINER" = "1" ] && rm -f "$LOCAL_CONTAINER"
