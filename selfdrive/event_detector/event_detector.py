#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
event-detector: 使用 openpilot 视觉模型对 Dashcam 视频进行在线处理与可视化

功能概述：
- 加载 openpilot 视觉模型（仅 vision 子网），从视频帧做预处理与推理；
- 支持从 JSON 内参文件读取 fisheye 参数，先去畸变再做几何归一化到模型标准视图；
- 针对“相机倒装（上下颠倒）”的场景，自动将帧旋转 180°，并对去畸变后的内参做相应修正；
- 一个简单的“在线标定”微调器：根据车道线中心偏移，对 yaw 做低通+微调，使投影更稳定；
- 将模型输出的车道线与前车（Lead）投影回当前帧进行实时叠加显示；

使用示例：
  python3 -m openpilot.selfdrive.event_detector.event_detector \
    --video ./data/07012025-1080P/video_1751374564005.mp4 \
    --intrinsics ./data/07012025-1080P/Intrinsic_matrix/FOV120-202504180014_1080p_fov0.522_params.json \
    --rotate180 --height 1.2

说明：
- 仅加载 vision 模型，输出解析复用 parse_model_outputs.Parser；
- 模型输入为两帧拼接的 YUV420 6 通道（共 12 通道），复用文档中定义的打包方式；
- 在线标定仅用于演示（小步长收敛 yaw），如需更稳健可扩展为利用路面/消失点估计；
"""

from __future__ import annotations

import os
import json
import time
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np

import os
# 统一 tinygrad 设备为 CPU，避免环境默认到 CUDA/AMD 导致 JIT 设备不一致
os.environ.setdefault('DEV', 'CPU')
os.environ.setdefault('DEVICE', 'CPU')
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.transformations.model import (
  get_warp_matrix,
  MEDMODEL_INPUT_SIZE,
)
from openpilot.common.transformations.camera import get_view_frame_from_road_frame


VISION_PKL_PATH = Path(__file__).resolve().parent.parent / 'modeld/models/driving_vision_tinygrad.pkl'
VISION_METADATA_PATH = Path(__file__).resolve().parent.parent / 'modeld/models/driving_vision_metadata.pkl'


@dataclass
class CamParams:
  # 畸变前（原始）
  K: np.ndarray               # 3x3
  D: np.ndarray               # 4 or 5 (fisheye)
  org_size: Tuple[int, int]   # (w,h)
  # 去畸变后的目标参数
  new_K: np.ndarray           # 3x3
  new_size: Tuple[int, int]   # (w,h)
  fisheye: bool               # 是否 fisheye 模型


def load_cam_params(json_path: str) -> CamParams:
  with open(json_path, 'r') as f:
    cfg = json.load(f)
  K = np.array(cfg['K'], dtype=np.float64).reshape(3, 3)
  D = np.array(cfg['D'], dtype=np.float64).reshape(-1)
  new_K = np.array(cfg.get('new_K', cfg['K']), dtype=np.float64).reshape(3, 3)
  org_w, org_h = cfg.get('org_image_size') or cfg.get('org_size') or cfg.get('image_size')
  new_size = tuple(cfg.get('new_image_size') or (org_w, org_h))
  typ = cfg.get('type', '').lower()
  fisheye = (typ == 'fisheye') or (len(D) == 4)
  return CamParams(K=K, D=D, org_size=(int(org_w), int(org_h)), new_K=new_K, new_size=(int(new_size[0]), int(new_size[1])), fisheye=fisheye)


def rotate_K_180(K: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
  # 对图像坐标 (u,v) 做 180° 旋转（绕图像中心），等价于像素翻转：u' = W-1-u, v' = H-1-v
  # 内参更新：fx, fy 不变；cx' = W-1-cx；cy' = H-1-cy
  W, H = size_wh
  Kp = K.copy()
  Kp[0, 2] = (W - 1.0) - K[0, 2]
  Kp[1, 2] = (H - 1.0) - K[1, 2]
  return Kp


def init_undistort_map(cam: CamParams) -> Tuple[np.ndarray, np.ndarray]:
  # 生成从“原始畸变图像 -> 去畸变新图像”的重映射表
  if cam.fisheye:
    K = cam.K.astype(np.float64)
    D = cam.D.astype(np.float64).reshape(-1, 1)
    new_K = cam.new_K.astype(np.float64)
    size = (int(cam.new_size[0]), int(cam.new_size[1]))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    return map1, map2
  else:
    map1, map2 = cv2.initUndistortRectifyMap(cam.K, cam.D, None, cam.new_K, cam.new_size, cv2.CV_16SC2)
    return map1, map2


def bgr_to_yuv420_6ch(img_bgr: np.ndarray) -> np.ndarray:
  # 输入：BGR uint8, 尺寸应为 (H=256, W=512)
  # 输出：6x128x256 的 YUV420 通道打包
  H, W = img_bgr.shape[:2]
  assert (W, H) == MEDMODEL_INPUT_SIZE, f"模型输入必须为 {MEDMODEL_INPUT_SIZE}, got {(W,H)}"
  # OpenCV 返回的 I420 是 (H*3/2, W) 的单通道平面，需要先展平后按字节数切片
  yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
  szY = H * W
  szU = (H // 2) * (W // 2)
  Y = yuv[:szY].reshape(H, W)
  U = yuv[szY:szY + szU].reshape(H // 2, W // 2)
  V = yuv[szY + szU:szY + szU + szU].reshape(H // 2, W // 2)
  ch0 = Y[::2, ::2]
  ch1 = Y[::2, 1::2]
  ch2 = Y[1::2, ::2]
  ch3 = Y[1::2, 1::2]
  chs = np.stack([ch0, ch1, ch2, ch3, U, V], axis=0).astype(np.uint8)
  return chs  # (6, 128, 256)


class OnlineYawCalibrator:
  """非常简化的在线标定：根据左右车道线中心偏移，缓慢调整 yaw

  - 目标：减少投影几何随时间的系统性偏移；
  - 方法：计算 (yR + yL)/2 的近端中心偏移 dy_center（道路系 y 左为正），以小增益逼近 0；
  - 注意：此方法仅演示用途，不保证严谨收敛。
  """
  def __init__(self, yaw_deg_init: float = 0.0, alpha: float = 0.05, gain: float = 0.001):
    self.yaw = np.deg2rad(yaw_deg_init)
    self.alpha = float(alpha)  # 一阶低通滤波系数
    self.gain = float(gain)    # yaw 微调增益（rad/m）
    self._ema_center = 0.0

  def update(self, lane_lines: np.ndarray) -> float:
    # lane_lines 形状: (1, 4, IDX_N, 2); 取内侧左右线 1/2 的 x=0 处 y 值
    try:
      yL = float(lane_lines[0, 1, 0, 0])
      yR = float(lane_lines[0, 2, 0, 0])
      center = 0.5 * (yL + yR)
      self._ema_center = (1.0 - self.alpha) * self._ema_center + self.alpha * center
      self.yaw += - self.gain * self._ema_center
    except Exception:
      pass
    return self.yaw


def project_road_points_to_img(xs: np.ndarray, ys: np.ndarray, K: np.ndarray, extrinsic_v_from_road: np.ndarray) -> np.ndarray:
  # xs, ys: 道路系坐标（米）；假设 z=0
  pts = np.stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)], axis=0)  # 4xN
  C = K @ extrinsic_v_from_road  # 3x4
  uvw = C @ pts  # 3xN
  uv = uvw[:2] / np.clip(uvw[2:3], 1e-6, None)
  return uv.T  # Nx2


def prepare_model_inputs(frame_ud_rot: np.ndarray, intrinsics_rot: np.ndarray, euler_device_from_calib: np.ndarray) -> Dict[str, Tensor]:
  # 为 'img' 与 'big_img' 生成 (1,12,128,256) 的 uint8 Tensor
  inputs = {}
  for name, big in (('img', False), ('big_img', True)):
    warp = get_warp_matrix(euler_device_from_calib.astype(np.float32), intrinsics_rot.astype(np.float32), bigmodel_frame=big)
    Minv = np.linalg.inv(warp)
    # warpPerspective 生成标准视图 (512x256)
    dst_size = MEDMODEL_INPUT_SIZE  # (w,h)
    warped = cv2.warpPerspective(frame_ud_rot, Minv, dst_size, flags=cv2.INTER_LINEAR)
    chs = bgr_to_yuv420_6ch(warped)
    # 维护两帧：若无历史帧则复制当前帧
    if not hasattr(prepare_model_inputs, '_prev_chs_'+name):
      setattr(prepare_model_inputs, '_prev_chs_'+name, chs)
    prev = getattr(prepare_model_inputs, '_prev_chs_'+name)
    combo = np.concatenate([prev, chs], axis=0)  # (12,128,256)
    setattr(prepare_model_inputs, '_prev_chs_'+name, chs)
    arr = combo[np.newaxis, ...]  # (1,12,128,256)
    inputs[name] = Tensor(arr.astype(np.uint8), dtype=dtypes.uint8).realize()
  return inputs


def draw_outputs(frame_show: np.ndarray, out: Dict[str, np.ndarray], K_rot: np.ndarray, height_m: float, euler_rpy: np.ndarray) -> None:
  # 投影并绘制车道线与 lead
  H, W = frame_show.shape[:2]
  v_from_road = get_view_frame_from_road_frame(0.0, float(euler_rpy[1]), float(euler_rpy[2]), float(height_m))
  xs = np.array(ModelConstants.X_IDXS, dtype=np.float32)

  # 车道线（取内侧左右 1,2）
  colorL = (0, 255, 0)
  colorR = (0, 200, 255)
  for idx, color in ((1, colorL), (2, colorR)):
    ys = out['lane_lines'][0, idx, :, 0].astype(np.float32)
    uv = project_road_points_to_img(xs, ys, K_rot, v_from_road)
    pts = []
    for u, v in uv:
      if 0 <= u < W and 0 <= v < H:
        pts.append((int(round(u)), int(round(v))))
    if len(pts) >= 2:
      cv2.polylines(frame_show, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=2)

  # 前车（取最可能假设 0，时刻 0）
  if 'lead' in out:
    try:
      lead0 = out['lead'][0, 0, 0]  # [x,y,v,a]
      x, y = float(lead0[0]), float(lead0[1])
      uv = project_road_points_to_img(np.array([x], dtype=np.float32), np.array([y], dtype=np.float32), K_rot, v_from_road)
      u, v = uv[0]
      if 0 <= u < W and 0 <= v < H:
        cv2.circle(frame_show, (int(round(u)), int(round(v))), 6, (0, 0, 255), 2)
        cv2.putText(frame_show, f"Lead x={x:.1f}m y={y:.1f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)
    except Exception:
      pass


def main():
  import argparse
  ap = argparse.ArgumentParser(description='openpilot event-detector (vision model on dashcam video)')
  ap.add_argument('--video', required=True, help='视频文件路径')
  ap.add_argument('--intrinsics', required=True, help='相机内参 JSON 文件路径')
  ap.add_argument('--rotate180', action='store_true', help='相机倒装：图像旋转 180 度并修正内参')
  ap.add_argument('--height', type=float, default=1.2, help='相机高度（米），用于投影显示')
  ap.add_argument('--yaw-init-deg', type=float, default=0.0, help='在线标定初始 yaw（度）')
  ap.add_argument('--save-out', type=str, default='', help='保存可视化到视频文件（mp4）；若不提供且无显示环境，将写入 /tmp/event_detector_out.mp4')
  ap.add_argument('--no-display', action='store_true', help='禁用窗口显示（适配 headless 环境）')
  ap.add_argument('--skip-seconds', type=float, default=0.0, help='启动时跳过视频前面指定秒数')
  args = ap.parse_args()

  # 加载模型
  if not VISION_PKL_PATH.exists() or not VISION_METADATA_PATH.exists():
    raise FileNotFoundError('找不到视觉模型或元数据，请确认 selfdrive/modeld/models/ 目录完整')
  with open(VISION_METADATA_PATH, 'rb') as f:
    import pickle
    md = pickle.load(f)
  vision_output_slices = md['output_slices']
  parser = Parser()
  with open(VISION_PKL_PATH, 'rb') as f:
    import pickle
    vision_run = pickle.load(f)

  # 加载相机模型与去畸变映射
  cam = load_cam_params(args.intrinsics)
  map1, map2 = init_undistort_map(cam)

  # 旋转后的去畸变内参（用于 warp 与投影）
  new_K_rot = rotate_K_180(cam.new_K, cam.new_size) if args.rotate180 else cam.new_K.copy()

  # 打开视频
  cap = cv2.VideoCapture(args.video)
  if not cap.isOpened():
    raise RuntimeError(f'无法打开视频文件: {args.video}')
  # 跳过视频前段
  if args.skip_seconds and args.skip_seconds > 0:
    # 优先用毫秒定位，失败再按帧定位
    if not cap.set(cv2.CAP_PROP_POS_MSEC, float(args.skip_seconds) * 1000.0):
      fps_try = cap.get(cv2.CAP_PROP_FPS)
      if fps_try and fps_try > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps_try * float(args.skip_seconds)))

  # 简易在线标定器（仅 yaw）
  calib = OnlineYawCalibrator(yaw_deg_init=args.yaw_init_deg, alpha=0.05, gain=0.001)

  stop = False
  def _signal(sig, frame):
    nonlocal stop
    stop = True
  signal.signal(signal.SIGINT, _signal)

  last_ts = time.time()
  # 显示/写出策略：尽量显示；若无 GUI 或出错则写文件
  display_env = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
  enable_display = (not args.no_display) and display_env and hasattr(cv2, 'imshow')
  writer = None
  out_path = args.save_out.strip()
  fps_src = cap.get(cv2.CAP_PROP_FPS)
  fps_out = fps_src if fps_src and fps_src > 0 else 20.0
  while not stop:
    ok, frame = cap.read()
    if not ok:
      # 循环播放
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      continue

    # 去畸变到 new_size
    frame_ud = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    # 倒装旋转
    if args.rotate180:
      frame_ud = cv2.rotate(frame_ud, cv2.ROTATE_180)

    # 生成模型输入（两路：img / big_img）
    euler = np.array([0.0, 0.0, calib.yaw], dtype=np.float32)
    model_inputs = prepare_model_inputs(frame_ud, new_K_rot, euler)

    # 前向推理
    outs = vision_run(**model_inputs).contiguous().realize().uop.base.buffer.numpy()
    # 切片并解析
    outs_dict = {k: outs[np.newaxis, v] for k, v in vision_output_slices.items()}
    parsed = parser.parse_vision_outputs(outs_dict)

    # 在线标定更新（根据车道中心）
    calib.update(parsed['lane_lines'])

    # 可视化叠加
    show = frame_ud.copy()
    draw_outputs(show, parsed, new_K_rot, args.height, np.array([0.0, 0.0, calib.yaw]))
    # 性能与状态文本
    now = time.time(); dt = now - last_ts; last_ts = now
    fps = 1.0 / max(dt, 1e-6)
    cv2.putText(show, f"FPS {fps:.1f}  yaw {np.rad2deg(calib.yaw):.2f} deg", (10, show.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # 输出：优先显示，否则写文件
    if enable_display and not out_path:
      try:
        cv2.imshow('event-detector', show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      except cv2.error:
        enable_display = False
        out_path = '/tmp/event_detector_out.mp4'

    if (not enable_display) or out_path:
      if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Hs, Ws = show.shape[:2]
        # 若用户未显式指定路径，默认到 /tmp
        if not out_path:
          out_path = '/tmp/event_detector_out.mp4'
        writer = cv2.VideoWriter(out_path, fourcc, float(fps_out), (Ws, Hs))
      writer.write(show)

  cap.release()
  try:
    if writer is not None:
      writer.release()
    if enable_display:
      cv2.destroyAllWindows()
  except cv2.error:
    pass


if __name__ == '__main__':
  main()
