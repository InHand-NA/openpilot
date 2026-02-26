"""Dataset for loading consecutive NPZ frame pairs.

Single-camera (DrivingDataset):
  - input: (12, 128, 256) float32 — 2 frames of YUV420 6-channel
  - targets: dict of GT tensors for all output heads

Dual-camera (DualCameraDrivingDataset):
  - img: (12, 128, 256) uint8 — road camera (medmodel warp)
  - big_img: (12, 128, 256) uint8 — wide camera (sbigmodel warp)
  - targets: dict of GT tensors

Frame pairing:
  current = sorted_npz_files[idx]
  previous = sorted_npz_files[max(0, idx - temporal_skip)]
  temporal_skip = MODEL_RUN_FREQ / MODEL_CONTEXT_FREQ = 20 / 5 = 4

Image preprocessing per frame:
  1. frame_rgb (1208, 1928, 3) uint8
  2. warp_image(rgb, rpyCalib, intrinsics) -> (256, 512, 3)
  3. rgb_to_yuv420_6ch(warped) -> (6, 128, 256) uint8
  4. Concatenate [prev_yuv, curr_yuv] -> (12, 128, 256)
  5. Single-cam: float32, /128.0 - 1.0 -> [-1, 1]
     Dual-cam: keep uint8, model normalizes internally
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, SBIGMODEL_INPUT_SIZE, calib_from_medmodel
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.modeld.constants import ModelConstants


TEMPORAL_SKIP = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ  # 4


def get_warp_matrix(rpyCalib: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
  """Compute 3x3 warp matrix from model input coords to camera pixel coords."""
  device_from_calib = rot_from_euler(rpyCalib)
  camera_from_calib = camera_intrinsics @ view_frame_from_device_frame @ device_from_calib
  return camera_from_calib @ calib_from_medmodel


def warp_image(rgb: np.ndarray, rpyCalib: np.ndarray, camera_intrinsics: np.ndarray) -> np.ndarray:
  """Warp raw camera image to model input space (512x256).

  Args:
    rgb: (H, W, 3) uint8 raw camera image
    rpyCalib: (3,) calibration euler angles (rad)
    camera_intrinsics: (3, 3) camera intrinsic matrix

  Returns:
    (256, 512, 3) uint8 warped image
  """
  M = get_warp_matrix(rpyCalib, camera_intrinsics)
  return cv2.warpPerspective(rgb, M, dsize=MEDMODEL_INPUT_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def rgb_to_yuv420_6ch(rgb: np.ndarray) -> np.ndarray:
  """Convert (256, 512, 3) uint8 RGB to (6, 128, 256) uint8 YUV420 6-channel.

  Channel layout: [y0, y1, u, y2, y3, v] matching loadyuv.cl.
  Uses BT.601 conversion matching rgb_to_nv12.cl.
  """
  R = rgb[..., 0].astype(np.int32)
  G = rgb[..., 1].astype(np.int32)
  B = rgb[..., 2].astype(np.int32)

  Y = np.clip(((13 * B + 65 * G + 33 * R + 64) >> 7) + 16, 0, 255)
  U = np.clip((56 * B - 37 * G - 19 * R + 0x8080) >> 8, 0, 255)
  V = np.clip((56 * R - 47 * G - 9 * B + 0x8080) >> 8, 0, 255)

  # Y subsampled into 4 channels of (128, 256)
  y0 = Y[0::2, 0::2]
  y1 = Y[1::2, 0::2]
  y2 = Y[0::2, 1::2]
  y3 = Y[1::2, 1::2]

  # UV 2x2 area average -> (128, 256)
  U_sub = (U[0::2, 0::2] + U[0::2, 1::2] + U[1::2, 0::2] + U[1::2, 1::2] + 2) >> 2
  V_sub = (V[0::2, 0::2] + V[0::2, 1::2] + V[1::2, 0::2] + V[1::2, 1::2] + 2) >> 2

  return np.stack([y0, y1, U_sub, y2, y3, V_sub], axis=0).astype(np.uint8)


def load_and_preprocess_frame(npz_data: dict, camera_intrinsics: np.ndarray) -> np.ndarray:
  """Load frame from npz dict, warp, and convert to YUV420 6-channel.

  Returns: (6, 128, 256) uint8
  """
  rgb = npz_data['frame_rgb']
  rpyCalib = npz_data['rpyCalib'].astype(np.float64)
  warped = warp_image(rgb, rpyCalib, camera_intrinsics)
  return rgb_to_yuv420_6ch(warped)


def extract_targets(npz_data: dict) -> dict[str, np.ndarray]:
  """Extract GT labels from npz data.

  Returns dict with:
    lane_lines: (4, 33, 2) — y, z only
    lane_lines_prob: (4,)
    road_edges: (2, 33, 2)
    road_edges_prob: (2,) — used for loss masking only
    lead: (3, 6, 4)
    lead_prob: (3,)
    pose: (6,)
    road_transform: (6,)
  """
  targets = {}

  # Lane lines: take y, z columns (indices 1, 2) from (4, 33, 3+)
  # NaN exists at points beyond hill crest (geometry occlusion cutoff)
  lane_lines = npz_data['lane_lines']
  ll = lane_lines[:, :, 1:3].astype(np.float32)  # (4, 33, 2)
  # Build per-point valid mask: (4, 33), True where no NaN in y or z
  targets['lane_lines_valid'] = (~np.isnan(ll).any(axis=-1)).astype(np.float32)
  targets['lane_lines'] = np.nan_to_num(ll, nan=0.0)
  targets['lane_lines_prob'] = npz_data['lane_lines_prob'].astype(np.float32)  # (4,)

  # Road edges: take y, z columns
  road_edges = npz_data['road_edges']
  re = road_edges[:, :, 1:3].astype(np.float32)  # (2, 33, 2)
  targets['road_edges_valid'] = (~np.isnan(re).any(axis=-1)).astype(np.float32)
  targets['road_edges'] = np.nan_to_num(re, nan=0.0)
  targets['road_edges_prob'] = npz_data['road_edges_prob'].astype(np.float32)  # (2,)

  # Lead
  targets['lead'] = npz_data['lead'].astype(np.float32)  # (3, 6, 4)
  targets['lead_prob'] = npz_data['lead_prob'].astype(np.float32)  # (3,)

  # Pose and road_transform
  targets['pose'] = npz_data['pose'].astype(np.float32)  # (6,)
  targets['road_transform'] = npz_data['road_transform'].astype(np.float32)  # (6,)

  return targets


class DrivingDataset(Dataset):
  """Dataset loading consecutive NPZ frame pairs.

  Args:
    data_dirs: list of directories containing NPZ files
    camera_type: device camera type key, default ("pc", "unknown")
  """

  def __init__(self, data_dirs: list[str], camera_type: tuple[str, str] = ("pc", "unknown")):
    self.camera_intrinsics = DEVICE_CAMERAS[camera_type].fcam.intrinsics

    # Collect and sort all NPZ files per directory, then flatten
    self.files: list[str] = []
    self.dir_boundaries: list[int] = []  # track directory boundaries for frame pairing

    for data_dir in sorted(data_dirs):
      data_path = Path(data_dir)
      npz_files = sorted(data_path.glob("*.npz"))
      if not npz_files:
        continue
      start_idx = len(self.files)
      self.dir_boundaries.append(start_idx)
      self.files.extend([str(f) for f in npz_files])

    if not self.files:
      raise ValueError(f"No NPZ files found in {data_dirs}")

    # Build set of directory start indices for efficient boundary check
    self._dir_starts = set(self.dir_boundaries)

  def __len__(self) -> int:
    return len(self.files)

  def _get_prev_idx(self, idx: int) -> int:
    """Get previous frame index respecting directory boundaries."""
    prev = idx - TEMPORAL_SKIP
    # Don't cross directory boundary
    for start in sorted(self._dir_starts, reverse=True):
      if start <= idx:
        return max(start, prev)
    return max(0, prev)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Load current and previous frames
    curr_data = dict(np.load(self.files[idx], allow_pickle=True))
    prev_idx = self._get_prev_idx(idx)
    prev_data = dict(np.load(self.files[prev_idx], allow_pickle=True))

    # Preprocess frames to YUV420 6-channel
    prev_yuv = load_and_preprocess_frame(prev_data, self.camera_intrinsics)
    curr_yuv = load_and_preprocess_frame(curr_data, self.camera_intrinsics)

    # Concatenate: (12, 128, 256)
    combined = np.concatenate([prev_yuv, curr_yuv], axis=0)

    # Normalize to [-1, 1]
    input_tensor = combined.astype(np.float32) / 128.0 - 1.0

    # Extract targets from current frame
    targets = extract_targets(curr_data)

    return (
      torch.from_numpy(input_tensor),
      {k: torch.from_numpy(v) for k, v in targets.items()},
    )


class CachedDualCameraDrivingDataset(Dataset):
  """Dataset loading pre-warped YUV420 cache files (fast dual-camera training).

  Cache files contain road_yuv (6,128,256) uint8 + wide_yuv (6,128,256) uint8
  + all GT labels — generated by preprocess_cache.py.

  Eliminates per-epoch warp + YUV conversion overhead.

  Args:
    cache_dirs: list of directories containing cache NPZ files
  """

  def __init__(self, cache_dirs: list[str]):
    self.files: list[str] = []
    self.dir_boundaries: list[int] = []

    for cache_dir in sorted(cache_dirs):
      cache_path = Path(cache_dir)
      npz_files = sorted(cache_path.glob('*.npz'))
      if not npz_files:
        continue
      start_idx = len(self.files)
      self.dir_boundaries.append(start_idx)
      self.files.extend([str(f) for f in npz_files])

    if not self.files:
      raise ValueError(f"No cache NPZ files found in {cache_dirs}")

    self._dir_starts = set(self.dir_boundaries)

  def __len__(self) -> int:
    return len(self.files)

  def _get_prev_idx(self, idx: int) -> int:
    prev = idx - TEMPORAL_SKIP
    for start in sorted(self._dir_starts, reverse=True):
      if start <= idx:
        return max(start, prev)
    return max(0, prev)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    curr = dict(np.load(self.files[idx], allow_pickle=True))
    prev_idx = self._get_prev_idx(idx)
    prev = dict(np.load(self.files[prev_idx], allow_pickle=True))

    # Stack [prev_yuv, curr_yuv] → (12, 128, 256) uint8
    road = np.concatenate([prev['road_yuv'], curr['road_yuv']], axis=0)
    wide = np.concatenate([prev['wide_yuv'], curr['wide_yuv']], axis=0)

    targets = {
      k: curr[k]
      for k in ('lane_lines', 'lane_lines_valid', 'lane_lines_prob',
                'road_edges', 'road_edges_valid', 'road_edges_prob',
                'lead', 'lead_prob', 'pose', 'road_transform')
    }

    return (
      torch.from_numpy(road),   # uint8 (12, 128, 256)
      torch.from_numpy(wide),   # uint8 (12, 128, 256)
      {k: torch.from_numpy(v) for k, v in targets.items()},
    )


class DualCameraDrivingDataset(Dataset):
  """Dataset loading dual-camera NPZ frame pairs (road_rgb + wide_rgb).

  Uses openpilot's get_warp_matrix for both cameras:
    - Road camera: medmodel warp (fl=910, bigmodel_frame=False)
    - Wide camera: sbigmodel warp (fl=455, bigmodel_frame=True)

  Outputs uint8 tensors — model handles normalization internally.

  Args:
    data_dirs: list of directories containing NPZ files (with road_rgb + wide_rgb)
    camera_type: device camera type key, default ("pc", "unknown")
  """

  def __init__(self, data_dirs: list[str], camera_type: tuple[str, str] = ("pc", "unknown")):
    dc = DEVICE_CAMERAS[camera_type]
    self.fcam_intrinsics = dc.fcam.intrinsics  # narrow camera
    self.ecam_intrinsics = dc.ecam.intrinsics  # wide camera

    # Collect and sort all NPZ files per directory
    self.files: list[str] = []
    self.dir_boundaries: list[int] = []

    for data_dir in sorted(data_dirs):
      data_path = Path(data_dir)
      npz_files = sorted(data_path.glob("*.npz"))
      if not npz_files:
        continue
      start_idx = len(self.files)
      self.dir_boundaries.append(start_idx)
      self.files.extend([str(f) for f in npz_files])

    if not self.files:
      raise ValueError(f"No NPZ files found in {data_dirs}")

    self._dir_starts = set(self.dir_boundaries)

  def __len__(self) -> int:
    return len(self.files)

  def _get_prev_idx(self, idx: int) -> int:
    """Get previous frame index respecting directory boundaries."""
    prev = idx - TEMPORAL_SKIP
    for start in sorted(self._dir_starts, reverse=True):
      if start <= idx:
        return max(start, prev)
    return max(0, prev)

  def _warp_road(self, rgb: np.ndarray, rpyCalib: np.ndarray) -> np.ndarray:
    """Road camera warp: medmodel (fl=910, bigmodel_frame=False)."""
    M = compute_warp_matrix(rpyCalib.astype(np.float64), self.fcam_intrinsics, bigmodel_frame=False)
    return cv2.warpPerspective(rgb, M, MEDMODEL_INPUT_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

  def _warp_wide(self, rgb: np.ndarray, rpyCalib: np.ndarray) -> np.ndarray:
    """Wide camera warp: sbigmodel (fl=455, bigmodel_frame=True)."""
    M = compute_warp_matrix(rpyCalib.astype(np.float64), self.ecam_intrinsics, bigmodel_frame=True)
    return cv2.warpPerspective(rgb, M, SBIGMODEL_INPUT_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    curr_data = dict(np.load(self.files[idx], allow_pickle=True))
    prev_idx = self._get_prev_idx(idx)
    prev_data = dict(np.load(self.files[prev_idx], allow_pickle=True))

    curr_rpyCalib = curr_data['rpyCalib'].astype(np.float64)
    prev_rpyCalib = prev_data['rpyCalib'].astype(np.float64)

    # Road camera: prev + curr → (12, 128, 256) uint8
    road = np.concatenate(
      [
        rgb_to_yuv420_6ch(self._warp_road(prev_data['road_rgb'], prev_rpyCalib)),
        rgb_to_yuv420_6ch(self._warp_road(curr_data['road_rgb'], curr_rpyCalib)),
      ],
      axis=0,
    )

    # Wide camera: prev + curr → (12, 128, 256) uint8
    wide = np.concatenate(
      [
        rgb_to_yuv420_6ch(self._warp_wide(prev_data['wide_rgb'], prev_rpyCalib)),
        rgb_to_yuv420_6ch(self._warp_wide(curr_data['wide_rgb'], curr_rpyCalib)),
      ],
      axis=0,
    )

    targets = extract_targets(curr_data)

    return (
      torch.from_numpy(road),  # uint8 (12, 128, 256)
      torch.from_numpy(wide),  # uint8 (12, 128, 256)
      {k: torch.from_numpy(v) for k, v in targets.items()},
    )
