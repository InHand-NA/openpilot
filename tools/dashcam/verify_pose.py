#!/usr/bin/env python3
"""离线验证 pose 和 road_transform 的准确性。

从 npz 序列加载数据，执行 4 项检查：
  1. pose[0] vs v_ego 一致性
  2. 轨迹积分 vs 世界坐标 (需要 world_pose 字段)
  3. road_transform[2] == camera_height
  4. calibrationd 式验证 (pose → rpyCalib)

用法:
  python tools/dashcam/verify_pose.py data/training/town04_xxx/
  python tools/dashcam/verify_pose.py data/training/town04_xxx/ --plot
"""

import argparse
import glob
import os
import sys

import numpy as np


def load_frames(clip_dir):
  """加载 npz 序列，返回按文件名排序的帧数据列表。"""
  pattern = os.path.join(clip_dir, "*.npz")
  files = sorted(glob.glob(pattern))
  if not files:
    print(f"错误：在 {clip_dir} 中未找到 .npz 文件")
    sys.exit(1)

  frames = []
  for f in files:
    data = dict(np.load(f, allow_pickle=True))
    frames.append(data)
  print(f"加载了 {len(frames)} 帧，来自 {clip_dir}")
  return frames


def check_pose_vs_vego(frames, dt=0.05):
  """检查 1：pose[0] (前向速度) vs v_ego (3D 速度标量)。"""
  pose_fwd = []
  v_egos = []
  for f in frames:
    pose_fwd.append(float(f['pose'][0]))
    v_egos.append(float(f['v_ego']))

  pose_fwd = np.array(pose_fwd)
  v_egos = np.array(v_egos)

  diff = pose_fwd - v_egos
  mae = np.mean(np.abs(diff))
  max_diff = np.max(np.abs(diff))
  corr = np.corrcoef(pose_fwd, v_egos)[0, 1] if len(pose_fwd) > 1 else 0.0

  print("\n=== 检查 1：pose[0] vs v_ego ===")
  print(f"  帧数:       {len(pose_fwd)}")
  print(f"  MAE:        {mae:.4f} m/s")
  print(f"  最大偏差:   {max_diff:.4f} m/s")
  print(f"  相关系数:   {corr:.6f}")
  print(f"  pose[0] 均值: {np.mean(pose_fwd):.3f} m/s")
  print(f"  v_ego 均值:   {np.mean(v_egos):.3f} m/s")

  return {'pose_fwd': pose_fwd, 'v_ego': v_egos, 'mae': mae, 'max_diff': max_diff, 'corr': corr}


def check_trajectory_vs_world(frames, dt=0.05):
  """检查 2：pose 积分轨迹 vs world_pose 世界坐标。"""
  # 检查 world_pose 是否可用
  if 'world_pose' not in frames[0]:
    print("\n=== 检查 2：轨迹积分 vs 世界坐标 ===")
    print("  world_pose not available，跳过此检查")
    return None

  # 提取世界坐标
  world_x = np.array([float(f['world_pose'][0]) for f in frames])
  world_y = np.array([float(f['world_pose'][1]) for f in frames])
  world_yaw_deg = np.array([float(f['world_pose'][5]) for f in frames])

  # 积分轨迹：从第一帧的世界坐标开始
  n = len(frames)
  int_x = np.zeros(n)
  int_y = np.zeros(n)
  int_x[0] = world_x[0]
  int_y[0] = world_y[0]

  # 使用第一帧的世界 yaw 作为初始航向角
  heading = np.deg2rad(world_yaw_deg[0])

  for i in range(1, n):
    pose = frames[i]['pose']
    v_fwd = float(pose[0])   # 前向速度 (calibrated frame)
    v_lat = float(pose[1])   # 侧向速度 (calibrated frame, 向右为正)
    yaw_rate = float(pose[5])  # yaw 角速度 (rad/s, 注意 pose 中的符号约定)

    # pose[5] 是 -d_yaw/dt (openpilot 约定)，所以实际 yaw 变化率取反
    heading += (-yaw_rate) * dt

    int_x[i] = int_x[i - 1] + (v_fwd * np.cos(heading) - v_lat * np.sin(heading)) * dt
    int_y[i] = int_y[i - 1] + (v_fwd * np.sin(heading) + v_lat * np.cos(heading)) * dt

  # 逐帧误差
  err_x = int_x - world_x
  err_y = int_y - world_y
  err_dist = np.sqrt(err_x**2 + err_y**2)

  # 终点误差
  end_err = err_dist[-1]

  # 滑动窗口积分误差（每 100 帧重置，避免长期漂移）
  window = 100
  window_errs = []
  for start in range(0, n - 1, window):
    end = min(start + window, n)
    if end - start < 2:
      continue
    # 从世界坐标重新开始积分
    w_int_x = world_x[start]
    w_int_y = world_y[start]
    w_heading = np.deg2rad(world_yaw_deg[start])
    for j in range(start + 1, end):
      pose = frames[j]['pose']
      v_fwd = float(pose[0])
      v_lat = float(pose[1])
      yaw_rate = float(pose[5])
      w_heading += (-yaw_rate) * dt
      w_int_x += (v_fwd * np.cos(w_heading) - v_lat * np.sin(w_heading)) * dt
      w_int_y += (v_fwd * np.sin(w_heading) + v_lat * np.cos(w_heading)) * dt
    w_err = np.sqrt((w_int_x - world_x[end - 1])**2 + (w_int_y - world_y[end - 1])**2)
    window_errs.append(w_err)

  mean_window_err = np.mean(window_errs) if window_errs else 0.0

  print("\n=== 检查 2：轨迹积分 vs 世界坐标 ===")
  print(f"  帧数:               {n}")
  print(f"  终点位移误差:       {end_err:.3f} m")
  print(f"  逐帧误差 (最大):   {np.max(err_dist):.3f} m")
  print(f"  逐帧误差 (均值):   {np.mean(err_dist):.3f} m")
  print(f"  滑动窗口误差 (w={window}, 均值): {mean_window_err:.3f} m")
  total_dist = np.sum(np.sqrt(np.diff(world_x)**2 + np.diff(world_y)**2))
  print(f"  总行驶距离:         {total_dist:.1f} m")
  if total_dist > 0:
    print(f"  终点误差/总距离:   {end_err / total_dist * 100:.2f}%")

  return {
    'int_x': int_x, 'int_y': int_y,
    'world_x': world_x, 'world_y': world_y,
    'err_dist': err_dist, 'end_err': end_err,
    'mean_window_err': mean_window_err,
  }


def check_road_transform(frames):
  """检查 3：road_transform[2] == camera_height。"""
  rt_heights = np.array([float(f['road_transform'][2]) for f in frames])
  cam_heights = np.array([float(f['camera_height']) for f in frames])

  diff = rt_heights - cam_heights
  mae = np.mean(np.abs(diff))
  max_diff = np.max(np.abs(diff))
  all_equal = np.allclose(rt_heights, cam_heights)

  print("\n=== 检查 3：road_transform[2] vs camera_height ===")
  print(f"  帧数:        {len(rt_heights)}")
  print(f"  完全一致:    {'是' if all_equal else '否'}")
  print(f"  MAE:         {mae:.6f} m")
  print(f"  最大偏差:    {max_diff:.6f} m")
  print(f"  rt[2] 均值:  {np.mean(rt_heights):.4f} m")
  print(f"  cam_h 均值:  {np.mean(cam_heights):.4f} m")

  return {'all_equal': all_equal, 'mae': mae}


def check_calib_from_pose(frames):
  """检查 4：校准坐标系一致性 — pose 速度方向应接近正前方。

  pose 在校准坐标系中，直行平路时应为 [v, 0, 0]。
  残差角 observed_rpy = [0, -atan2(pose[2], pose[0]), atan2(pose[1], pose[0])]
  应接近 0。若偏离过大，说明 pose 标注与 rpyCalib 不匹配。
  """
  pitch_residual_list = []
  yaw_residual_list = []
  v_egos = []

  for f in frames:
    pose = f['pose']
    v_fwd = float(pose[0])
    v_lat = float(pose[1])
    v_vert = float(pose[2])
    v_ego = float(f['v_ego'])
    v_egos.append(v_ego)

    if v_ego > 5.0:
      # 校准坐标系下，速度方向偏离正前方的残差角
      pitch_residual_list.append(-np.arctan2(v_vert, v_fwd))
      yaw_residual_list.append(np.arctan2(v_lat, v_fwd))

  pitch_res = np.array(pitch_residual_list)
  yaw_res = np.array(yaw_residual_list)

  n_valid = len(pitch_res)

  print("\n=== 检查 4：校准坐标系一致性 (残差角应接近 0) ===")
  print(f"  有效帧数 (v_ego > 5 m/s):  {n_valid} / {len(frames)}")

  if n_valid == 0:
    print("  无有效帧，跳过")
    return None

  print("  --- pitch 残差 (-atan2(pose[2], pose[0])) ---")
  print(f"    均值:   {np.degrees(np.mean(pitch_res)):.4f}°")
  print(f"    MAE:    {np.degrees(np.mean(np.abs(pitch_res))):.4f}°")
  print(f"    标准差: {np.degrees(np.std(pitch_res)):.4f}°")
  print(f"    最大值: {np.degrees(np.max(np.abs(pitch_res))):.4f}°")
  print("  --- yaw 残差 (atan2(pose[1], pose[0])) ---")
  print(f"    均值:   {np.degrees(np.mean(yaw_res)):.4f}°")
  print(f"    MAE:    {np.degrees(np.mean(np.abs(yaw_res))):.4f}°")
  print(f"    标准差: {np.degrees(np.std(yaw_res)):.4f}°")
  print(f"    最大值: {np.degrees(np.max(np.abs(yaw_res))):.4f}°")

  return {
    'pitch_res': pitch_res, 'yaw_res': yaw_res,
    'v_egos': np.array(v_egos),
  }


def plot_results(results, output_dir):
  """生成验证图表并保存为 PNG。"""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))

  # 子图 1：pose[0] vs v_ego
  ax = axes[0, 0]
  r1 = results['check1']
  n = len(r1['pose_fwd'])
  t = np.arange(n) * 0.05
  ax.plot(t, r1['pose_fwd'], label='pose[0]', alpha=0.8)
  ax.plot(t, r1['v_ego'], label='v_ego', alpha=0.8, linestyle='--')
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Speed (m/s)')
  ax.set_title(f'pose[0] vs v_ego (MAE={r1["mae"]:.4f})')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # 子图 2：积分轨迹 vs 世界坐标
  ax = axes[0, 1]
  r2 = results.get('check2')
  if r2 is not None:
    ax.plot(r2['world_x'], r2['world_y'], label='world_pose (GT)', linewidth=2)
    ax.plot(r2['int_x'], r2['int_y'], label='Integrated', linestyle='--', linewidth=1.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'BEV Trajectory (end_err={r2["end_err"]:.2f}m)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
  else:
    ax.text(0.5, 0.5, 'world_pose\nnot available', ha='center', va='center', fontsize=14)
    ax.set_title('BEV Trajectory (no data)')

  # 子图 3：pitch 残差时序
  ax = axes[1, 0]
  r4 = results.get('check4')
  if r4 is not None:
    n_valid = len(r4['pitch_res'])
    idx = np.arange(n_valid)
    ax.plot(idx, np.degrees(r4['pitch_res']), alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='ideal (0)')
    ax.set_xlabel('Valid Frame Index')
    ax.set_ylabel('Residual (deg)')
    ax.set_title(f'Pitch Residual (MAE={np.degrees(np.mean(np.abs(r4["pitch_res"]))):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
  else:
    ax.text(0.5, 0.5, 'No valid frames\n(v_ego > 5 m/s)', ha='center', va='center', fontsize=14)
    ax.set_title('Pitch Residual (no data)')

  # 子图 4：yaw 残差时序
  ax = axes[1, 1]
  if r4 is not None:
    ax.plot(idx, np.degrees(r4['yaw_res']), alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='ideal (0)')
    ax.set_xlabel('Valid Frame Index')
    ax.set_ylabel('Residual (deg)')
    ax.set_title(f'Yaw Residual (MAE={np.degrees(np.mean(np.abs(r4["yaw_res"]))):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
  else:
    ax.text(0.5, 0.5, 'No valid frames\n(v_ego > 5 m/s)', ha='center', va='center', fontsize=14)
    ax.set_title('Yaw Residual (no data)')

  plt.tight_layout()
  out_path = os.path.join(output_dir, "pose_verify.png")
  plt.savefig(out_path, dpi=150)
  plt.close()
  print(f"\n图表已保存: {out_path}")


def main():
  parser = argparse.ArgumentParser(description='离线验证 pose 和 road_transform')
  parser.add_argument('clip_dir', help='npz 数据目录路径')
  parser.add_argument('--plot', action='store_true', help='生成 matplotlib 图表')
  parser.add_argument('--dt', type=float, default=0.05, help='帧间时间间隔 (默认 0.05s = 20Hz)')
  args = parser.parse_args()

  if not os.path.isdir(args.clip_dir):
    print(f"错误：目录不存在: {args.clip_dir}")
    sys.exit(1)

  frames = load_frames(args.clip_dir)

  results = {}
  results['check1'] = check_pose_vs_vego(frames, dt=args.dt)
  results['check2'] = check_trajectory_vs_world(frames, dt=args.dt)
  results['check3'] = check_road_transform(frames)
  results['check4'] = check_calib_from_pose(frames)

  print("\n=== 验证完成 ===")

  if args.plot:
    plot_results(results, args.clip_dir)


if __name__ == "__main__":
  main()
