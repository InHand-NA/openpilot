#!/usr/bin/env python3
"""示例：演示训练时如何对原始图像做 warp 矫正。

原始图像在设备坐标系（有安装偏差），模型期望输入在校准坐标系（正前方无偏）。
训练时需要用 rpyCalib 做透视变换，将原始图像 warp 到模型输入空间。

warp 链路：
  原始相机图像 (1928x1208)
      ↓  camera_intrinsics⁻¹
  归一化设备坐标
      ↓  device_from_calib (rpyCalib 旋转)
  校准坐标
      ↓  model_intrinsics
  模型输入图像 (512x256)

用法：
  python tools/dashcam/warp_example.py data/training/test_h1.6-12/000200.npz
  python tools/dashcam/warp_example.py data/training/test_h1.6-12/000200.npz --no-calib  # 不做校准（rpyCalib=0）
"""

import argparse
import os
import sys

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import (
  MEDMODEL_INPUT_SIZE,
  calib_from_medmodel,
  medmodel_intrinsics,
)
from openpilot.common.transformations.orientation import rot_from_euler


def get_warp_matrix(rpyCalib, camera_intrinsics):
  """计算从模型输入坐标到原始相机像素坐标的 3x3 warp 矩阵。

  与 openpilot 的 get_warp_matrix() 相同：
    warp = camera_intrinsics @ view_from_device @ device_from_calib @ calib_from_model

  Args:
    rpyCalib: [3] 校准欧拉角 [roll, pitch, yaw] (弧度)
    camera_intrinsics: [3,3] 相机内参矩阵

  Returns:
    [3,3] warp 矩阵，用于 cv2.warpPerspective
  """
  device_from_calib = rot_from_euler(rpyCalib)
  camera_from_calib = camera_intrinsics @ view_frame_from_device_frame @ device_from_calib
  return camera_from_calib @ calib_from_medmodel


def warp_image(rgb, rpyCalib, camera_intrinsics):
  """将原始相机图像 warp 到模型输入空间 (512x256)。

  Args:
    rgb: [H, W, 3] uint8 原始相机图像
    rpyCalib: [3] 校准欧拉角 (弧度)
    camera_intrinsics: [3,3] 相机内参矩阵

  Returns:
    [256, 512, 3] uint8 warp 后的图像
  """
  M = get_warp_matrix(rpyCalib, camera_intrinsics)
  # warp_matrix 是从模型坐标到相机坐标的映射
  # warpPerspective 需要的是 dst→src 映射，所以直接用 M
  warped = cv2.warpPerspective(
    rgb, M,
    dsize=MEDMODEL_INPUT_SIZE,  # (512, 256)
    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
  )
  return warped


def main():
  parser = argparse.ArgumentParser(description='Warp 示例')
  parser.add_argument('npz_path', help='.npz 文件路径')
  parser.add_argument('--no-calib', action='store_true',
                      help='不做校准矫正 (rpyCalib=[0,0,0])，模拟 calibrationd 尚未收敛的情况')
  args = parser.parse_args()

  if not os.path.exists(args.npz_path):
    print(f"文件不存在: {args.npz_path}")
    sys.exit(1)

  data = dict(np.load(args.npz_path, allow_pickle=True))

  rgb = data['frame_rgb']
  rpyCalib = data['rpyCalib'].astype(np.float64)
  cam_pitch = float(data['camera_pitch'])
  cam_yaw = float(data['camera_yaw'])

  # PC/simulator 使用 ecam 内参 (与 Carla dashcam 的 road-only 模式一致)
  dc = DEVICE_CAMERAS[("pc", "unknown")]
  camera_intrinsics = dc.fcam.intrinsics

  print(f"原始图像: {rgb.shape}")
  print(f"安装角: pitch={np.degrees(cam_pitch):.1f}° yaw={np.degrees(cam_yaw):.1f}°")
  print(f"rpyCalib: [{np.degrees(rpyCalib[0]):.2f}, {np.degrees(rpyCalib[1]):.2f}, {np.degrees(rpyCalib[2]):.2f}]°")
  print(f"模型输入: {MEDMODEL_INPUT_SIZE}")
  print(f"模型焦距: {medmodel_intrinsics[0,0]:.0f}")
  print(f"相机焦距: {camera_intrinsics[0,0]:.0f}")

  # 1. 正确校准：用 rpyCalib warp
  warped_correct = warp_image(rgb, rpyCalib, camera_intrinsics)

  # 2. 无校准：rpyCalib = [0,0,0]
  warped_no_calib = warp_image(rgb, np.zeros(3), camera_intrinsics)

  # 3. 错误校准：rpyCalib 偏差 50%
  rpyCalib_wrong = rpyCalib * 0.5
  warped_wrong = warp_image(rgb, rpyCalib_wrong, camera_intrinsics)

  # 绘图
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(2, 2, figsize=(14, 8))

  # 原始图像 (缩小显示)
  ax = axes[0, 0]
  ax.imshow(rgb)
  h = rgb.shape[0]
  ax.axhline(y=h // 2, color='r', linestyle='--', alpha=0.5)
  ax.set_title(f'Raw Camera (pitch={np.degrees(cam_pitch):.1f}, yaw={np.degrees(cam_yaw):.1f})')

  # 正确校准
  ax = axes[0, 1]
  ax.imshow(warped_correct)
  h_model = MEDMODEL_INPUT_SIZE[1]
  ax.axhline(y=h_model // 2, color='r', linestyle='--', alpha=0.5)
  ax.set_title('Correct Calib (rpyCalib from GT)')

  # 无校准
  ax = axes[1, 0]
  ax.imshow(warped_no_calib)
  ax.axhline(y=h_model // 2, color='r', linestyle='--', alpha=0.5)
  ax.set_title('No Calib (rpyCalib=[0,0,0])')

  # 错误校准
  ax = axes[1, 1]
  ax.imshow(warped_wrong)
  ax.axhline(y=h_model // 2, color='r', linestyle='--', alpha=0.5)
  ax.set_title('Wrong Calib (rpyCalib * 0.5)')

  plt.suptitle('Training Image Warp: raw -> model input (512x256)', fontsize=14)
  plt.tight_layout()

  out_dir = os.path.dirname(args.npz_path)
  out_path = os.path.join(out_dir, "warp_example.png")
  plt.savefig(out_path, dpi=150)
  plt.close()
  print(f"\n图表已保存: {out_path}")


if __name__ == "__main__":
  main()
