#!/bin/bash
set -e

# Minimal COLMAP reconstruction - handheld only, default settings
CONTAINER="/tmp/colmap_$$.sif"
cp /cluster/research-groups/wehrwein/zoo/containers/colmap.sif "$CONTAINER"
colmap() {
    apptainer exec --nv "$CONTAINER" colmap "$@"
}

rm -f database.db
rm -rf sparse_test
mkdir -p sparse_test

echo "=== Feature extraction ==="
colmap feature_extractor \
    --database_path database.db \
    --image_path ./handheld

echo "=== Exhaustive matching ==="
colmap exhaustive_matcher \
    --database_path database.db

echo "=== Mapper ==="
colmap mapper \
    --database_path database.db \
    --image_path ./handheld \
    --output_path sparse_test

echo "=== Results ==="
for d in sparse_test/*/; do
    echo "Model: $d"
    colmap model_analyzer --path "$d"
done

rm -f "$CONTAINER"
