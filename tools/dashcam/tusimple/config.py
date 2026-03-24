"""TuSimple data collection configuration — camera params, height definitions, crop params.

All camera intrinsics, height constants, and TuSimple format constants live here.
Phase 1~4 scripts import from this single source of truth.
"""

from dataclasses import dataclass

import numpy as np

# ── H0: openpilot reference stereo camera (annotation baseline) ──────────────
H0_HEIGHT = 1.22             # mount height (meters)
H0_W, H0_H = 1928, 1208
H0_NARROW_FOV = 40           # degrees
H0_WIDE_FOV = 120            # degrees
H0_NARROW_FOCAL = 2648.0
H0_WIDE_FOCAL = 567.0

# ── H1~H6: Mono single-eye camera (Carla pinhole rendering) ─────────────────
MONO_W, MONO_H = 1920, 1080
MONO_HFOV = 120  # degrees
MONO_FOCAL = MONO_W / 2 / np.tan(np.radians(MONO_HFOV / 2))  # ≈554.256

K_MONO = np.array([
  [MONO_FOCAL, 0.0,        MONO_W / 2],
  [0.0,        MONO_FOCAL, MONO_H / 2],
  [0.0,        0.0,        1.0],
])

# ── TuSimple output ─────────────────────────────────────────────────────────
TUSIMPLE_W, TUSIMPLE_H = 1280, 720

# TuSimple standard h_samples: v=160 to v=710, step 10
TUSIMPLE_H_SAMPLES = list(range(160, 720, 10))  # 56 sample points

# ── ROI crop parameters (120° → target FOV, improve far-field resolution) ────
CROP_HFOV = 70        # degrees, effective HFOV after crop
NOMINAL_PITCH = 4.0   # degrees, nominal mount pitch (fixes crop region)
HORIZON_RATIO = 0.3   # horizon position in cropped image (0=top, 1=bottom)


def compute_crop_params(
  pitch_deg: float = NOMINAL_PITCH,
  crop_hfov: float = CROP_HFOV,
  horizon_ratio: float = HORIZON_RATIO,
) -> dict:
  """Compute ROI crop parameters and equivalent intrinsics.

  Crops a center region of crop_hfov from 1920x1080 (HFOV=120°),
  maintaining 16:9 aspect ratio, then scales to 1280x720.

  Uses fixed NOMINAL_PITCH to compute crop region — shared by training
  and deployment. Different session pitches cause the horizon to float
  naturally in the cropped image, improving model robustness.

  Returns:
    {
      'crop_rect': (x, y, w, h),    # crop rectangle in 1920×1080
      'crop_hfov': float,            # effective HFOV after crop (degrees)
      'K_crop': np.ndarray,          # equivalent 3×3 intrinsics for 1280×720
      'effective_focal': float,      # equivalent focal length at TuSimple res
    }
  """
  # Crop size (pixels)
  crop_w = int(2 * MONO_FOCAL * np.tan(np.radians(crop_hfov / 2)))
  crop_h = crop_w * TUSIMPLE_H // TUSIMPLE_W   # maintain 16:9

  # Horizontal center
  crop_x = (MONO_W - crop_w) // 2

  # Vertical: place horizon at horizon_ratio using nominal pitch
  horizon_y = MONO_H / 2 - MONO_FOCAL * np.tan(np.radians(pitch_deg))
  crop_y = int(horizon_y - horizon_ratio * crop_h)
  crop_y = max(0, min(crop_y, MONO_H - crop_h))

  # Equivalent intrinsics (crop region → 1280×720)
  scale = TUSIMPLE_W / crop_w
  f_crop = MONO_FOCAL * scale
  cx_crop = (MONO_W / 2 - crop_x) * scale    # = TUSIMPLE_W / 2 when centered
  cy_crop = (MONO_H / 2 - crop_y) * scale

  K_crop = np.array([
    [f_crop, 0.0,    cx_crop],
    [0.0,    f_crop, cy_crop],
    [0.0,    0.0,    1.0],
  ])

  return {
    'crop_rect': (crop_x, crop_y, crop_w, crop_h),
    'crop_hfov': crop_hfov,
    'K_crop': K_crop,
    'effective_focal': f_crop,
  }

# Fixed crop parameters (NOMINAL_PITCH=4°, CROP_HFOV=70°):
#   crop_rect = (572, 370, 776, 436)
#   K_crop = [[914.3, 0, 640.0], [0, 914.3, 280.4], [0, 0, 1]]
#   effective_focal = 914.3


# ── Height definitions ───────────────────────────────────────────────────────
HEIGHT_DEFS = {
  'H1': 1.22,   # standard sedan (same height as H0)
  'H2': 1.30,
  'H3': 1.50,   # mid-size SUV
  'H4': 2.00,
  'H5': 2.50,
  'H6': 3.00,   # truck cab
}


@dataclass
class CameraSlotConfig:
  tag: str       # 'H1', ..., 'H6'
  height: float  # meters above ground
