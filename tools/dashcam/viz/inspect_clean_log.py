#!/usr/bin/env python3
"""逐帧浏览 clean_log.txt 中选中的帧及其标注。

遍历清洗结果，显示 warp 后的图像 + 标注叠加（车道线/路沿/前车/BEV）。
绘制风格复用 inspect_annotated.py。

Keys:
  Left / Right   - prev / next frame
  PgUp / PgDn    - ±10 frames
  Home / End     - first / last frame
  Tab            - switch narrow / wide camera
  h              - cycle heights  H1 → H2 → … → H6 → H1
  d              - toggle disable/enable current frame
  l              - toggle lane lines
  e              - toggle road edges
  v              - toggle lead vehicles
  b              - toggle BEV panel
  i              - toggle info panel
  r              - toggle raw / warped image
  s              - screenshot
  q / ESC        - quit

用法：
  python tools/dashcam/viz/inspect_clean_log.py \\
      --dataset-dir /nfs/openpilot-datasets/multi_height-0312/ \\
      --clean-log data/multi_height-0312/clean_log.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.model import get_warp_matrix as compute_warp_matrix
from openpilot.common.transformations.orientation import rot_from_euler

# Reuse all drawing helpers from inspect_annotated
from openpilot.tools.dashcam.viz.inspect_annotated import (
  BEV_W,
  DISPLAY_H,
  DISPLAY_W,
  MODEL_H,
  MODEL_W,
  _load_json_annotation,
  build_display_transform,
  draw_bev,
  draw_lane_lines,
  draw_lead_prob_bar,
  draw_leads,
  draw_prob_bar,
  draw_road_edges,
)


KEY_LEFT  = 65361
KEY_RIGHT = 65363
KEY_PGUP  = 65365
KEY_PGDN  = 65366
KEY_HOME  = 65360
KEY_END   = 65367
KEY_TAB   = 9

HEIGHTS = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']


# ---------------------------------------------------------------------------
# clean_log 解析
# ---------------------------------------------------------------------------

def load_clean_log(path: Path) -> list[tuple[str, int]]:
  """读取 clean_log.txt → [(session, frame_id), ...]"""
  entries = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      session, fid_str = line.rsplit('/', 1)
      entries.append((session, int(fid_str)))
  return entries


def entry_key(session: str, frame_id: int) -> str:
  return f'{session}/{frame_id:06d}'


def load_disabled_frames(path: Path) -> set[str]:
  """读取 disabled_frames.txt → set of 'session/frame_id'"""
  if not path.exists():
    return set()
  with open(path) as f:
    return {line.strip() for line in f if line.strip()}


def save_disabled_frames(path: Path, disabled: set[str]) -> None:
  """保存 disabled_frames.txt（排序后写入）"""
  with open(path, 'w') as f:
    for entry in sorted(disabled):
      f.write(entry + '\n')


# ---------------------------------------------------------------------------
# Session 缓存（避免每帧重复计算 warp 矩阵）
# ---------------------------------------------------------------------------

class SessionCache:
  """缓存 session 级数据：clip_info, warp 矩阵, intrinsics"""

  def __init__(self, dataset_dir: Path):
    self._dataset_dir = dataset_dir
    self._cache: dict[str, dict] = {}

  def get(self, session: str) -> dict:
    if session in self._cache:
      return self._cache[session]

    clip_path = self._dataset_dir / session / 'clip_info.json'
    with open(clip_path) as f:
      clip_info = json.load(f)

    pitch_rad = math.radians(clip_info['camera']['pitch_deg'])
    yaw_rad   = math.radians(clip_info['camera']['yaw_deg'])
    rpyCalib  = np.array([0.0, pitch_rad, yaw_rad], dtype=np.float64)

    dc = DEVICE_CAMERAS[('pc', 'unknown')]
    warp_road = compute_warp_matrix(rpyCalib, dc.fcam.intrinsics, bigmodel_frame=False)
    warp_wide = compute_warp_matrix(rpyCalib, dc.ecam.intrinsics, bigmodel_frame=True)

    entry = {
      'clip_info': clip_info,
      'rpyCalib': rpyCalib,
      'warp_road': warp_road,
      'warp_wide': warp_wide,
      'K_road': dc.fcam.intrinsics,
      'K_wide': dc.ecam.intrinsics,
      'heights': clip_info.get('heights', {}),
    }
    self._cache[session] = entry
    return entry


# ---------------------------------------------------------------------------
# 帧渲染
# ---------------------------------------------------------------------------

def render_frame(
  data: dict,
  session: str,
  frame_id: int,
  entry_idx: int,
  total: int,
  height_tag: str,
  height_m: float,
  rpyCalib: np.ndarray,
  warp_road: np.ndarray,
  warp_wide: np.ndarray,
  K_road: np.ndarray,
  K_wide: np.ndarray,
  cam_idx: int,
  show_lanes: bool,
  show_edges: bool,
  show_leads: bool,
  show_bev: bool,
  show_info: bool,
  show_raw: bool,
  is_disabled: bool = False,
  n_disabled: int = 0,
  estimate_lead_z: bool = True,
) -> np.ndarray:

  # Select camera
  if cam_idx == 0:
    rgb_key, warp, K = 'road_rgb', warp_road, K_road
    cam_label = 'NARROW'
  else:
    rgb_key, warp, K = 'wide_rgb', warp_wide, K_wide
    cam_label = 'WIDE'

  rgb = data.get(rgb_key, data.get('road_rgb'))
  if rgb is None:
    img = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(img, "No image", (DISPLAY_W // 3, DISPLAY_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
  elif show_raw:
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (DISPLAY_W, DISPLAY_H))
  else:
    warped = cv2.warpPerspective(rgb, warp, (MODEL_W, MODEL_H),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    img = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR), (DISPLAY_W, DISPLAY_H))

  # Build projection matrix
  T_disp = build_display_transform(warp, K, rpyCalib)

  # Annotation overlays
  camera_height = float(data.get('camera_height', height_m))
  if not show_raw:
    if show_lanes:
      draw_lane_lines(img, data, T_disp)
    if show_edges:
      draw_road_edges(img, data, T_disp)
    if show_leads:
      draw_leads(img, data, T_disp, camera_height, estimate_z=estimate_lead_z)

  # Camera + height label (top-right)
  label = f"{cam_label}  {height_tag} {height_m:.2f}m"
  (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
  lx = img.shape[1] - tw - 14
  cv2.rectangle(img, (lx - 4, 0), (img.shape[1], 30), (0, 0, 0), -1)
  cv2.putText(img, label, (lx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

  # DISABLED banner (red bar across top)
  if is_disabled:
    banner_h = 36
    roi = img[0:banner_h, :]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], banner_h), (0, 0, 180), -1)
    img[0:banner_h, :] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
    cv2.putText(img, "DISABLED  (press 'd' to re-enable)", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

  # Info panel
  if show_info:
    ll_prob   = data.get('lane_lines_prob', np.zeros(4))
    lead_prob = data.get('lead_prob', np.zeros(3))
    v_ego     = float(data.get('v_ego', 0.0))
    pitch_d   = math.degrees(rpyCalib[1])
    yaw_d     = math.degrees(rpyCalib[2])

    lines = [
      (f"[{entry_idx+1}/{total}] {session}/{frame_id:06d}", (200, 200, 200)),
      (f"Speed: {v_ego:.1f} m/s  ({v_ego * 3.6:.1f} km/h)", (255, 255, 255)),
      (f"pitch={pitch_d:+.2f}deg  yaw={yaw_d:+.2f}deg  h={camera_height:.2f}m",
       (255, 255, 255)),
      (f"LL prob: {ll_prob[0]:.2f}  {ll_prob[1]:.2f}  {ll_prob[2]:.2f}  {ll_prob[3]:.2f}",
       (255, 255, 255)),
      (f"Lead:  {lead_prob[0]:.2f}  {lead_prob[1]:.2f}  {lead_prob[2]:.2f}",
       (255, 255, 255)),
      (f"{'[raw]' if show_raw else '[warped]'}  disabled: {n_disabled}",
       (180, 180, 180)),
    ]
    info_y0 = 40 if is_disabled else 0
    dy, pw = 26, 520
    ph = dy * len(lines) + 12
    ph = min(ph, img.shape[0] - info_y0)
    pw = min(pw, img.shape[1])
    roi     = img[info_y0:info_y0 + ph, 0:pw]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (pw, ph), (0, 0, 0), -1)
    img[info_y0:info_y0 + ph, 0:pw] = cv2.addWeighted(overlay, 0.60, roi, 0.40, 0)
    y = info_y0 + 22
    for text, color in lines:
      cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
      y += dy

  # Probability bars at bottom
  ll_prob   = data.get('lane_lines_prob', np.zeros(4))
  lead_prob = data.get('lead_prob', np.zeros(3))
  prob_bar = np.zeros((30, DISPLAY_W, 3), dtype=np.uint8)
  draw_prob_bar(prob_bar, ll_prob, 1, 0.05)
  lead_bar = np.zeros((30, DISPLAY_W, 3), dtype=np.uint8)
  draw_lead_prob_bar(lead_bar, lead_prob, 1)
  main = np.vstack([img, prob_bar, lead_bar])

  # BEV panel
  if show_bev:
    bev = np.zeros((main.shape[0], BEV_W, 3), dtype=np.uint8)
    draw_bev(bev, data)
    main = np.hstack([main, bev])

  return main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description='浏览 clean_log.txt 中的清洗帧')
  parser.add_argument('--dataset-dir', type=Path, required=True,
                      help='原始数据集根目录（包含 Town* sessions）')
  parser.add_argument('--clean-log', type=Path, required=True,
                      help='clean_log.txt 文件路径')
  parser.add_argument('--height', default='H1',
                      help='初始显示高度 (default: H1)')
  parser.add_argument('--start', type=int, default=0,
                      help='起始帧索引 (default: 0)')
  parser.add_argument('--no-lead-z', action='store_true',
                      help='禁用 lead z 插值估算，使用相机安装高度')
  args = parser.parse_args()

  dataset_dir = args.dataset_dir.resolve()
  if not dataset_dir.exists():
    print(f"ERROR: dataset_dir not found: {dataset_dir}", file=sys.stderr)
    sys.exit(1)

  entries = load_clean_log(args.clean_log)
  if not entries:
    print("ERROR: clean_log.txt is empty", file=sys.stderr)
    sys.exit(1)

  total = len(entries)
  cache = SessionCache(dataset_dir)

  # disabled_frames.txt 与 clean_log.txt 同目录
  disabled_path = args.clean_log.parent / 'disabled_frames.txt'
  disabled = load_disabled_frames(disabled_path)

  height_tag = args.height
  height_idx = HEIGHTS.index(height_tag) if height_tag in HEIGHTS else 0

  idx         = max(0, min(args.start, total - 1))
  cam_idx     = 0
  show_lanes  = True
  show_edges  = True
  show_leads  = True
  show_bev    = True
  show_info   = True
  show_raw    = False

  print(f"clean_log: {args.clean_log} ({total} frames)")
  print(f"disabled:  {disabled_path} ({len(disabled)} frames)")
  print(f"dataset:   {dataset_dir}")
  print(f"height:    {height_tag}")
  print("Keys: \u2190\u2192 frames  PgUp/PgDn \u00b110  Tab=cam  h=height  d=disable  "
        "l/e/v=overlays  b=BEV  i=info  r=raw  s=screenshot  q=quit")

  win = 'inspect_clean_log'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win, DISPLAY_W + BEV_W, DISPLAY_H + 60)

  while True:
    session, frame_id = entries[idx]
    sc = cache.get(session)
    height_m = sc['heights'].get(height_tag, 1.22)

    # Load annotation
    ann_path = dataset_dir / session / 'annotations' / height_tag / f'{frame_id:06d}.json'
    if ann_path.exists():
      data = _load_json_annotation(ann_path)
    else:
      data = {}

    # Load images
    img_dir = dataset_dir / session / height_tag
    road_bgr = cv2.imread(str(img_dir / f'road_{frame_id:06d}.png'))
    wide_bgr = cv2.imread(str(img_dir / f'wide_{frame_id:06d}.png'))
    if road_bgr is not None:
      data['road_rgb'] = cv2.cvtColor(road_bgr, cv2.COLOR_BGR2RGB)
    if wide_bgr is not None:
      data['wide_rgb'] = cv2.cvtColor(wide_bgr, cv2.COLOR_BGR2RGB)

    cur_key = entry_key(session, frame_id)
    cur_disabled = cur_key in disabled

    img = render_frame(
      data=data,
      session=session,
      frame_id=frame_id,
      entry_idx=idx,
      total=total,
      height_tag=height_tag,
      height_m=height_m,
      rpyCalib=sc['rpyCalib'],
      warp_road=sc['warp_road'],
      warp_wide=sc['warp_wide'],
      K_road=sc['K_road'],
      K_wide=sc['K_wide'],
      cam_idx=cam_idx,
      show_lanes=show_lanes,
      show_edges=show_edges,
      show_leads=show_leads,
      show_bev=show_bev,
      show_info=show_info,
      show_raw=show_raw,
      is_disabled=cur_disabled,
      n_disabled=len(disabled),
      estimate_lead_z=not args.no_lead_z,
    )

    cam_name = 'NARROW' if cam_idx == 0 else 'WIDE'
    cv2.setWindowTitle(win,
      f"[{idx+1}/{total}] {session}/{frame_id:06d}  "
      f"{height_tag} {height_m:.2f}m  [{cam_name}]")
    cv2.imshow(win, img)

    key = cv2.waitKeyEx(0)
    if key in (ord('q'), 27):
      break
    elif key == KEY_RIGHT:
      idx = min(idx + 1, total - 1)
    elif key == KEY_LEFT:
      idx = max(idx - 1, 0)
    elif key == KEY_PGDN:
      idx = min(idx + 10, total - 1)
    elif key == KEY_PGUP:
      idx = max(idx - 10, 0)
    elif key == KEY_HOME:
      idx = 0
    elif key == KEY_END:
      idx = total - 1
    elif key == KEY_TAB:
      cam_idx = 1 - cam_idx
      print(f"Camera: {'WIDE' if cam_idx else 'NARROW'}")
    elif key == ord('d'):
      if cur_key in disabled:
        disabled.discard(cur_key)
        print(f"ENABLED:  {cur_key}  (disabled: {len(disabled)})")
      else:
        disabled.add(cur_key)
        print(f"DISABLED: {cur_key}  (disabled: {len(disabled)})")
      save_disabled_frames(disabled_path, disabled)
    elif key == ord('h'):
      height_idx = (height_idx + 1) % len(HEIGHTS)
      height_tag = HEIGHTS[height_idx]
      print(f"Height: {height_tag}")
    elif key == ord('l'):
      show_lanes = not show_lanes
    elif key == ord('e'):
      show_edges = not show_edges
    elif key == ord('v'):
      show_leads = not show_leads
    elif key == ord('b'):
      show_bev = not show_bev
    elif key == ord('i'):
      show_info = not show_info
    elif key == ord('r'):
      show_raw = not show_raw
    elif key == ord('s'):
      path = f"screenshot_clean_{session.replace('/', '_')}_{height_tag}_{frame_id:06d}.png"
      cv2.imwrite(path, img)
      print(f"Screenshot: {path}")

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
