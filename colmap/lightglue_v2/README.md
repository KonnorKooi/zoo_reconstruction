# LightGlue v2 -- Automated Stationary Camera Registration (Not Used)

Alternative approach to registering stationary cameras that replaces manual 2D-3D correspondences with automated SuperPoint + LightGlue feature matching. This method was explored but not used in the final pipeline. The manual registration approach in `../manual_registration/` was used instead.

## How It Works

1. Match stationary images to handheld images using SuperPoint features and LightGlue matching.
2. Project known 3D points from the handheld reconstruction onto handheld images.
3. Bridge SuperPoint matches to 3D points by finding the nearest projected point to each matched keypoint.
4. Collect 2D (stationary pixel) to 3D (world point) correspondences.
5. Solve PnP + RANSAC to estimate the stationary camera pose directly.

## Files

| File | Purpose |
|---|---|
| `register_stationary.py` | SuperPoint/LightGlue matching and direct PnP registration |
| `pipeline_stationary_cameras.sh` | End-to-end pipeline (handheld COLMAP then LightGlue PnP) |
| `colmap_lightglue.job` | SLURM job script |

## Key Parameters

The pipeline script passes several tuning parameters to the registration script:

- `--max_keypoint_dist 10` -- max pixel distance when bridging matches to projected 3D points
- `--min_correspondences 12` -- minimum 2D-3D pairs needed to attempt PnP
- `--pnp_reproj_threshold 4.0` -- RANSAC reprojection error threshold in pixels
- `--multiscale` with `--scales "0.25,0.5,0.75,1.0,1.5,2.0,3.0"` -- run SuperPoint at multiple scales for more feature coverage

## Dependencies

Requires `lightglue` (SuperPoint + LightGlue) and PyTorch in addition to the standard COLMAP dependencies.
