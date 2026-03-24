"""3D→2D 投影核心算法库，供 project_tusimple.py 调用。

坐标系:
  openpilot calibration frame: x=forward, y=right, z=down
  投影: K @ view_frame_from_device_frame @ rot_from_euler(rpyCalib) @ point_3d
"""

import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

from openpilot.tools.dashcam.tusimple.config import TUSIMPLE_W, TUSIMPLE_H_SAMPLES


def project_3d_to_mono(
  points_3d: np.ndarray,
  K: np.ndarray,
  rpyCalib: np.ndarray,
) -> np.ndarray:
  """将标定坐标系中的 3D 点投影到相机像素坐标。

  与 visualizer.py:project_points_to_image() 完全一致。
  相机后方的点 (proj[2] <= 0) 返回 NaN。

  Args:
    points_3d: (N, 3) [x, y, z] in calibration frame
    K: 3×3 相机内参
    rpyCalib: [roll, pitch, yaw] 弧度

  Returns:
    (N, 2) float array, [u, v] 像素坐标, 无效点为 NaN
  """
  device_from_calib = rot_from_euler(rpyCalib)
  T = K @ view_frame_from_device_frame @ device_from_calib
  pts = points_3d.T  # (3, N)
  proj = T @ pts      # (3, N)
  behind = proj[2] <= 0
  proj[2, behind] = np.nan
  uv = (proj[:2] / proj[2:3]).T  # (N, 2)
  return uv


def resample_lane_at_h_samples(
  uv: np.ndarray,
  h_samples: list[int],
  x_range: tuple[int, int] = (0, TUSIMPLE_W - 1),
) -> list[int]:
  """在 TuSimple h_samples 的每个 v 值处插值 x 坐标。

  前置条件: 输入 v 已经过单调性裁断。

  Returns:
    长度为 len(h_samples) 的整数列表, -2 表示不可见
  """
  valid = ~np.isnan(uv).any(axis=1)
  u, v = uv[valid, 0], uv[valid, 1]

  if len(v) < 2:
    return [-2] * len(h_samples)

  # 按 v 升序排列 (供 np.interp)
  idx = np.argsort(v)
  v_sorted, u_sorted = v[idx], u[idx]

  # 去除重复 v 值
  unique_mask = np.diff(v_sorted, prepend=-1.0) > 0
  v_sorted = v_sorted[unique_mask]
  u_sorted = u_sorted[unique_mask]

  if len(v_sorted) < 2:
    return [-2] * len(h_samples)

  lane_x: list[int] = []
  for h in h_samples:
    if h < v_sorted[0] or h > v_sorted[-1]:
      lane_x.append(-2)
    else:
      x_interp = float(np.interp(h, v_sorted, u_sorted))
      x_int = int(round(x_interp))
      if x_range[0] <= x_int <= x_range[1]:
        lane_x.append(x_int)
      else:
        lane_x.append(-2)

  return lane_x


def _monotone_cutoff(v: np.ndarray) -> int:
  """找到 v 单调递减的截断点 (v 应从大到小)。返回保留的长度。"""
  for j in range(1, len(v)):
    if v[j] >= v[j - 1]:
      return j
  return len(v)


def lanes_3d_to_tusimple(
  lane_lines_3d: np.ndarray,
  lane_lines_prob: np.ndarray,
  road_edges_3d: np.ndarray,
  K_tusimple: np.ndarray,
  rpyCalib: np.ndarray,
  h_samples: list[int] | None = None,
  lane_prob_threshold: float = 0.2,
  min_visible_pts: int = 2,
  min_x_distance: float = 2.0,
) -> list[list[int]]:
  """完整的 3D 车道线到 TuSimple 2D 格式转换。

  车道线选择策略 (TuSimple 固定 4 条):
    slot 0 (L-out): lane[0] 优先, 低置信 → road_edge[0] 补位
    slot 1 (L-inn): lane[1] only
    slot 2 (R-inn): lane[2] only
    slot 3 (R-out): lane[3] 优先, 低置信 → road_edge[1] 补位

  Returns:
    4 条车道线列表, 每条是 len(h_samples) 的整数列表
  """
  if h_samples is None:
    h_samples = TUSIMPLE_H_SAMPLES

  # 构建 4 个槽位的 3D 源: (pts_3d, prob)
  slots: list[tuple[np.ndarray, float]] = []
  for i in range(4):
    prob = float(lane_lines_prob[i])
    if prob > lane_prob_threshold:
      slots.append((lane_lines_3d[i], prob))
    elif i == 0 and road_edges_3d.shape[0] > 0:
      # L-out 补位: road_edge[0]
      slots.append((road_edges_3d[0], 1.0))
    elif i == 3 and road_edges_3d.shape[0] > 1:
      # R-out 补位: road_edge[1]
      slots.append((road_edges_3d[1], 1.0))
    else:
      slots.append((None, 0.0))

  lanes: list[list[int]] = []
  for pts_3d, prob in slots:
    if pts_3d is None or prob <= 0:
      lanes.append([-2] * len(h_samples))
      continue

    # 过滤近场
    mask = pts_3d[:, 0] >= min_x_distance
    pts = pts_3d[mask]
    if pts.shape[0] < 2:
      lanes.append([-2] * len(h_samples))
      continue

    # 投影到 TuSimple 分辨率
    uv = project_3d_to_mono(pts, K_tusimple, rpyCalib)

    # 滤除无效
    valid = ~np.isnan(uv).any(axis=1)
    u, v = uv[valid, 0], uv[valid, 1]
    if len(v) < 2:
      lanes.append([-2] * len(h_samples))
      continue

    # 单调性裁断: 近→远, v 应从大(底)到小(顶) 单调递减
    cutoff = _monotone_cutoff(v)
    u, v = u[:cutoff], v[:cutoff]
    if len(v) < 2:
      lanes.append([-2] * len(h_samples))
      continue

    # 重组为 uv 数组，调用 resample
    uv_cut = np.stack([u, v], axis=1)
    lane_x = resample_lane_at_h_samples(uv_cut, h_samples)

    # 检查最少可见点
    n_visible = sum(1 for x in lane_x if x != -2)
    if n_visible < min_visible_pts:
      lanes.append([-2] * len(h_samples))
    else:
      lanes.append(lane_x)

  return lanes
