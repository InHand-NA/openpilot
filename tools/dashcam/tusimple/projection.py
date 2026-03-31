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
) -> tuple[list[list[int]], list[str]]:
  """完整的 3D 车道线到 TuSimple 2D 格式转换。

  车道线选择策略 (TuSimple 固定 4 条):
    每条 road_edge 最多补位一个槽位, 内侧线优先。
    左侧: road_edge[0] 优先补 L-inn(slot 1), 否则补 L-out(slot 0)
    右侧: road_edge[1] 优先补 R-inn(slot 2), 否则补 R-out(slot 3)

  车道线名称:
    LL  = 左外车道线 (lane_lines[0])
    L   = 左内车道线 (lane_lines[1], 当前车道左边界)
    R   = 右内车道线 (lane_lines[2], 当前车道右边界)
    RR  = 右外车道线 (lane_lines[3])
    LRE = 左路沿 (road_edges[0], 补位 L 或 LL)
    RRE = 右路沿 (road_edges[1], 补位 R 或 RR)

  Returns:
    (lanes, lane_names):
      lanes: 4 条车道线列表, 每条是 len(h_samples) 的整数列表
      lane_names: 4 条车道线的名称列表
  """
  if h_samples is None:
    h_samples = TUSIMPLE_H_SAMPLES

  # 槽位名称: lane_lines 原始名 vs road_edge 补位名
  SLOT_LANE_NAMES = ['LL', 'L', 'R', 'RR']

  # Road edge 补位决策: 每条 road_edge 最多补一个槽位，内侧优先
  # 左侧: road_edge[0] 优先补 L-inn(slot 1), 否则补 L-out(slot 0)
  # 右侧: road_edge[1] 优先补 R-inn(slot 2), 否则补 R-out(slot 3)
  left_re_slot = -1   # road_edge[0] 补位到哪个 slot, -1 表示未分配
  right_re_slot = -1  # road_edge[1] 补位到哪个 slot
  if road_edges_3d.shape[0] > 0:
    l_in_prob = float(lane_lines_prob[1])
    l_out_prob = float(lane_lines_prob[0])
    if l_in_prob <= lane_prob_threshold:
      left_re_slot = 1   # 补位 L-inn
    elif l_out_prob <= lane_prob_threshold:
      left_re_slot = 0   # 补位 L-out
  if road_edges_3d.shape[0] > 1:
    r_in_prob = float(lane_lines_prob[2])
    r_out_prob = float(lane_lines_prob[3])
    if r_in_prob <= lane_prob_threshold:
      right_re_slot = 2  # 补位 R-inn
    elif r_out_prob <= lane_prob_threshold:
      right_re_slot = 3  # 补位 R-out

  # 构建 4 个槽位的 3D 源: (pts_3d, prob, name)
  slots: list[tuple[np.ndarray, float, str]] = []
  for i in range(4):
    prob = float(lane_lines_prob[i])
    if prob > lane_prob_threshold:
      slots.append((lane_lines_3d[i], prob, SLOT_LANE_NAMES[i]))
    elif i == left_re_slot:
      slots.append((road_edges_3d[0], 1.0, 'LRE'))
    elif i == right_re_slot:
      slots.append((road_edges_3d[1], 1.0, 'RRE'))
    else:
      slots.append((None, 0.0, SLOT_LANE_NAMES[i]))

  lanes: list[list[int]] = []
  lane_names: list[str] = []
  for pts_3d, prob, name in slots:
    if pts_3d is None or prob <= 0:
      lanes.append([-2] * len(h_samples))
      lane_names.append(name)
      continue

    # 过滤近场
    mask = pts_3d[:, 0] >= min_x_distance
    pts = pts_3d[mask]
    if pts.shape[0] < 2:
      lanes.append([-2] * len(h_samples))
      lane_names.append(name)
      continue

    # 投影到 TuSimple 分辨率
    uv = project_3d_to_mono(pts, K_tusimple, rpyCalib)

    # 滤除无效
    valid = ~np.isnan(uv).any(axis=1)
    u, v = uv[valid, 0], uv[valid, 1]
    if len(v) < 2:
      lanes.append([-2] * len(h_samples))
      lane_names.append(name)
      continue

    # 单调性裁断: 近→远, v 应从大(底)到小(顶) 单调递减
    cutoff = _monotone_cutoff(v)
    u, v = u[:cutoff], v[:cutoff]
    if len(v) < 2:
      lanes.append([-2] * len(h_samples))
      lane_names.append(name)
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
    lane_names.append(name)

  return lanes, lane_names
