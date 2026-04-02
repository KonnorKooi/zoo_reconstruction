# Zoo 3D Reconstruction 

This repo includes 3D reconstruction pipelines, animal tracking pipelines, and custom april tag creation and detection.

## Folders

### `colmap/`

COLMAP-based 3D reconstruction pipelines. Builds sparse and dense models from handheld video, then registers stationary zoo cameras into the reconstruction.

The best pipeline used to place the stationary web cameras in the scene effectively was plain colmap sparse -> manual_registration -> dense (with high density settings, manually done through UI on local computer) 

- **`manual_registration/`** -- Primary pipeline. Two-stage approach: COLMAP sparse reconstruction on handheld images, then PnP registration of stationary cameras using manually-clicked 2D-3D correspondences. Includes optional dense reconstruction (fused point cloud + Poisson mesh).
- **`lightglue_v2/`** -- Alternative automated registration using SuperPoint + LightGlue feature matching instead of manual correspondences. Explored but not used in the final pipeline.
- **`lightglue_v1/`**, **`stationary/`** -- Earlier iterations of the registration approach.

### `rhino-tracking/`

Tracks a rhino's 3D position in video by casting rays from 2D bounding box detections through a calibrated camera into the dense mesh produced by the COLMAP pipeline. Uses Rerun for live 3D visualization.

### `tags/`

AprilTag fiducial marker pipeline. Generates printable tag sheets, resizes camera images, detects tags, and estimates camera poses in COLMAP format. Used for camera pose estimation via known tag positions in the environment.
