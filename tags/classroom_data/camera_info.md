# Camera Information — Classroom Data

## Hardware

| Field | Value |
|---|---|
| Camera | Canon EOS 5D Mark IV |
| Lens | Canon EF 24-105mm f/4L IS II USM |
| Sensor | Full-frame CMOS, 36×24mm |
| Native resolution | 6720×4480 px |
| Pixel pitch | 4.440 μm (X) / 4.432 μm (Y) |

---

## Focal Lengths Used Across Dataset

**20 of 21 images shot at 24mm. One image (8Y0A8436.JPG) shot at 31mm.**
The pipeline must use per-image intrinsics — a single shared K matrix will be wrong for 8Y0A8436.JPG.

---

## Camera Intrinsics (from EXIF Focal Plane Resolution)

Pixel density derived from EXIF fields:
- `Focal Plane X Resolution`: 5719.148936 px/inch → **225.163 px/mm**
- `Focal Plane Y Resolution`: 5728.900256 px/inch → **225.547 px/mm**

Formula: `fx = focal_length_mm × px_per_mm`

### At native resolution (6720×4480)

| Focal Length | fx | fy | cx | cy |
|---|---|---|---|---|
| 24mm | 5403.92 | 5413.13 | 3360.0 | 2240.0 |
| 31mm | 6980.06 | 6991.96 | 3360.0 | 2240.0 |

### At resized resolution (1000×667) — used by pipeline

Scale factors: `sx = 1000/6720 = 0.14881`, `sy = 667/4480 = 0.14888`

| Focal Length | fx | fy | cx | cy |
|---|---|---|---|---|
| 24mm | 804.15 | 805.93 | 500.0 | 333.5 |
| 31mm | 1038.70 | 1040.99 | 500.0 | 333.5 |

---

## Lens Distortion

`dist = np.zeros(5)` is currently used — **no calibration has been performed**.

The Canon EF 24-105mm f/4L IS II USM has measurable barrel distortion, particularly at 24mm.
Note: `Digital Lens Optimizer` was **OFF** during capture, so in-camera correction was not applied.

**To do:** Run `cv2.calibrateCamera()` with a checkerboard or ChArUco board at both 24mm and 31mm.
Until then, pose estimates carry distortion-induced error — translation accuracy will be lower than if calibrated.

---

## Other Capture Settings

| Field | Value |
|---|---|
| Exposure | Manual (1/250s) |
| Aperture | f/5.0 |
| ISO | 12800 (auto base) |
| Shutter | Mechanical |

High ISO (12800) may introduce noise — relevant if AprilTag detection struggles on dark/textured areas.
