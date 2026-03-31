# COLMAP Reconstruction Pipeline -- Manual Stationary Camera Registration

Two-stage pipeline that builds a 3D reconstruction from handheld video, then registers stationary zoo cameras into it using manually-clicked 2D-3D correspondences.

## Overview

**Stage 1** -- Standard COLMAP sparse reconstruction on handheld images only (feature extraction, exhaustive matching, mapping, bundle adjustment).

**Stage 2** -- Register stationary cameras by solving PnP from manual 2D-3D correspondences, then run a final bundle adjustment over all cameras together.

An optional dense reconstruction step produces a fused point cloud and Poisson surface mesh.

## Directory Structure

```
manual_registration/
├── pipeline_manual.sh           # Full sparse pipeline (Stage 1 + Stage 2)
├── pipeline_dense.sh            # Dense reconstruction (stereo + fusion + meshing)
├── register_manual.py           # PnP registration of stationary cameras
├── annotate_points.py           # GUI for clicking 2D-3D correspondences
├── visualize_reprojections.py   # Validate poses by reprojecting 3D points
├── correspondences.json         # Manual 2D-3D point pairs per stationary image
├── colmap_manual.job            # SLURM job script for sparse pipeline
├── colmap_dense.job             # SLURM job script for dense pipeline
├── fused.ply                    # Dense point cloud output
├── meshed-poisson.ply           # Poisson surface mesh output
└── output/                      # Pipeline outputs
```

## Workflow

### 1. Run the sparse pipeline

`pipeline_manual.sh` runs both stages end to end. It uses an Apptainer container for COLMAP binaries.

```bash
bash pipeline_manual.sh
```

This produces:
- `01_handheld/` -- COLMAP sparse model from handheld images only
- `02_with_stationary/sparse/registered/` -- model with stationary cameras added via PnP
- `02_with_stationary/sparse/registered_bin/` -- same, in binary format
- `02_with_stationary/sparse/optimized/` -- final model after joint bundle adjustment

### 2. Annotate correspondences (if adding new stationary cameras)

Use the GUI to click landmarks on a stationary image and type their known 3D coordinates.

```bash
uv run annotate_points.py
```

Saves to `correspondences.json`:
```json
{
  "image_name.jpg": {
    "points_2d": [[x, y], ...],
    "points_3d": [[X, Y, Z], ...]
  }
}
```

3D coordinates come from known points in the handheld reconstruction (e.g., identifiable features whose 3D position you can read from the COLMAP viewer).

### 3. Register stationary cameras

```bash
uv run register_manual.py
```

Reads the handheld COLMAP model and `correspondences.json`, solves PnP+RANSAC for each stationary camera, and writes the extended model in text format.

### 4. Validate

Reproject the 3D correspondences onto the registered images and check alignment:

```bash
uv run visualize_reprojections.py
```

### 5. Dense reconstruction (optional)

After the sparse model is finalized:

```bash
bash pipeline_dense.sh
```

Produces `fused.ply` (dense point cloud) and `meshed-poisson.ply` (surface mesh). The mesh is used downstream by the rhino tracking ray-mesh intersection pipeline.

## COLMAP Configuration

- Feature extraction: SIFT, 16384 max features
- Matching: exhaustive
- Camera models: SIMPLE_PINHOLE and PINHOLE supported
- Runs via Apptainer container (path configured in the pipeline scripts)

## Dependencies

Python side: matplotlib, numpy, opencv-python (see `pyproject.toml`).
COLMAP binaries: provided via Apptainer container.
