#!/usr/bin/env python3
"""Multi-process dashcam: Bridge (Carla + VisionIPC + visualization) + modeld subprocess + optional calibrationd.

Usage:
  # Start Carla server first, then:
  python tools/dashcam/run.py --perfect-cam --high-quality
  python tools/dashcam/run.py --online-calib --camera-pitch 5 --camera-yaw 3
  python tools/dashcam/run.py --perfect-cam --save-video output.mp4 --max-frames 100
"""

import argparse
import os
import signal
import subprocess
import sys
import time

# Environment variables must be set before importing openpilot modules
os.environ['NOBOARD'] = '1'
os.environ['SIMULATION'] = '1'
os.environ.setdefault('PYOPENCL_CTX', '')  # auto-select OpenCL platform, avoid interactive prompt

import numpy as np

from cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.hardware import HARDWARE
from openpilot.tools.dashcam.camerad import DashcamCamerad

TICKS_PER_FRAME = 2  # 2 ticks × 0.025s = 0.05s per frame = 20 FPS


def publish_device_state(pm):
  """Publish deviceState so modeld can determine DEVICE_CAMERAS config."""
  dat = messaging.new_message('deviceState', valid=True)
  dat.deviceState.deviceType = HARDWARE.get_device_type()
  pm.send('deviceState', dat)


def publish_car_state(pm, v_ego):
  """Publish carState with vehicle speed (calibrationd uses vEgo)."""
  dat = messaging.new_message('carState', valid=True)
  dat.carState.vEgo = float(v_ego)
  pm.send('carState', dat)


def publish_live_calibration(pm, rpyCalib, height):
  """Known pose mode: publish fixed calibration values."""
  msg = messaging.new_message('liveCalibration', valid=True)
  msg.liveCalibration.rpyCalib = rpyCalib.tolist()
  msg.liveCalibration.validBlocks = 20
  msg.liveCalibration.calStatus = log.LiveCalibrationData.Status.calibrated
  msg.liveCalibration.height = [height]
  pm.send('liveCalibration', msg)


def main():
  parser = argparse.ArgumentParser(description='Multi-process dashcam with openpilot vision')
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--town', default='Town04_Opt')
  parser.add_argument('--spawn-point', type=int, default=16)
  parser.add_argument('--camera-pitch', type=float, default=5.0,
                      help='Camera pitch in degrees')
  parser.add_argument('--camera-yaw', type=float, default=3.0,
                      help='Camera yaw in degrees')
  parser.add_argument('--camera-height', type=float, default=1.13,
                      help='Camera height in meters')
  parser.add_argument('--perfect-cam', action='store_true',
                      help='Use pitch=0, yaw=0 (ideal mounting)')
  parser.add_argument('--online-calib', action='store_true',
                      help='Use online calibration (calibrationd subprocess)')
  parser.add_argument('--num-npc', type=int, default=20)
  parser.add_argument('--high-quality', action='store_true')
  parser.add_argument('--no-display', action='store_true')
  parser.add_argument('--save-video', type=str, default='',
                      help='Save visualization to mp4 file')
  parser.add_argument('--max-frames', type=int, default=0,
                      help='Stop after N frames (0 = unlimited)')
  parser.add_argument('--fast', action='store_true',
                      help='Run as fast as possible, bypass frame rate limiter')
  parser.add_argument('--wide-road-only', action='store_true',
                      help='Single wide camera mode (modeld uses ecam intrinsics for both inputs)')
  parser.add_argument('--road-only', action='store_true',
                      help='Single narrow camera mode (modeld uses fcam intrinsics for main input)')
  args = parser.parse_args()

  if args.wide_road_only and args.road_only:
    parser.error('--wide-road-only and --road-only are mutually exclusive')

  # Camera pose
  if args.perfect_cam:
    pitch_deg, yaw_deg = 0.0, 0.0
  else:
    pitch_deg, yaw_deg = args.camera_pitch, args.camera_yaw

  camera_height = args.camera_height
  # rpyCalib convention: [roll, -pitch, -yaw] in radians
  rpyCalib = np.array([0.0, -np.deg2rad(pitch_deg), -np.deg2rad(yaw_deg)])

  print(f"Camera pose: pitch={pitch_deg}\u00b0 yaw={yaw_deg}\u00b0 height={camera_height}m")
  if args.wide_road_only:
    cam_mode_str = 'wide-road-only (single ecam)'
  elif args.road_only:
    cam_mode_str = 'road-only (single fcam)'
  else:
    cam_mode_str = 'dual (fcam + ecam)'
  print(f"Camera mode: {cam_mode_str}")
  print(f"Calibration mode: {'online (calibrationd)' if args.online_calib else 'known pose'}")

  # 1. Initialize Params
  params = Params()

  # Write CarParams (modeld and calibrationd block on this)
  from opendbc.car.car_helpers import get_demo_car_params
  CP = get_demo_car_params()
  params.put("CarParams", CP.to_bytes())

  # Write CalibrationParams (calibrationd reads this on startup)
  calib_msg = messaging.new_message('liveCalibration')
  if args.online_calib:
    calib_msg.liveCalibration.validBlocks = 0
    calib_msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  else:
    calib_msg.liveCalibration.validBlocks = 20
    calib_msg.liveCalibration.rpyCalib = rpyCalib.tolist()
  params.put("CalibrationParams", calib_msg.to_bytes())

  # 2. Create VisionIPC server (must be before starting modeld)
  print("Creating VisionIPC server...")
  camerad = DashcamCamerad(wide_road_only=args.wide_road_only, road_only=args.road_only)

  # 3. Start modeld subprocess (use CUDA on NVIDIA GPU for faster inference)
  # CUDA backend handles FP16 natively; CL has exp2(half) ambiguity on NVIDIA OpenCL
  modeld_env = {**os.environ, 'DEV': 'CUDA', 'PYOPENCL_CTX': ''}
  print(f"Starting modeld subprocess (DEV={modeld_env['DEV']})...")
  modeld_proc = subprocess.Popen(
    [sys.executable, '-m', 'selfdrive.modeld.modeld'],
    env=modeld_env)

  # 4. Optionally start calibrationd subprocess
  calibrationd_proc = None
  if args.online_calib:
    print("Starting calibrationd subprocess...")
    calibrationd_proc = subprocess.Popen(
      [sys.executable, '-m', 'openpilot.tools.dashcam.calibrationd'],
      env={**os.environ})

  # 5. Create cereal pub/sub
  pub_services = ['carState', 'deviceState']
  if not args.online_calib:
    pub_services.append('liveCalibration')
  pm = messaging.PubMaster(pub_services)
  sm = messaging.SubMaster(['modelV2', 'liveCalibration'])

  # 6. Connect to Carla
  print("Connecting to Carla...")
  from openpilot.tools.dashcam.carla_world import DashcamCarlaWorld
  world = DashcamCarlaWorld(
    host=args.host, port=args.port, town=args.town,
    spawn_point=args.spawn_point,
    camera_pitch_deg=pitch_deg, camera_yaw_deg=yaw_deg,
    camera_height=camera_height,
    high_quality=args.high_quality, num_npc=args.num_npc,
    wide_road_only=args.wide_road_only, road_only=args.road_only)

  # Camera intrinsics (for visualization)
  # wide-road-only: modeld uses ecam.intrinsics for both transforms, so visualization must match
  # road-only / dual: display the narrow road camera, use fcam.intrinsics
  dc = DEVICE_CAMERAS[("pc", "unknown")]
  vis_intrinsics = dc.ecam.intrinsics if args.wide_road_only else dc.fcam.intrinsics

  # 7. Initialize visualizer
  from openpilot.tools.dashcam.visualizer import Visualizer
  visualizer = Visualizer(
    save_video_path=args.save_video,
    no_display=args.no_display,
    source_fps=20.0)

  # Signal handler
  running = True
  def signal_handler(sig, frame):
    nonlocal running
    if not running:
      sys.exit(1)  # Force exit on second signal
    running = False
    print("\nShutting down...")
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  # Main loop
  TARGET_FPS = 20.0
  FRAME_DT = 1.0 / TARGET_FPS  # 50ms per frame
  print("Starting main loop...")
  tick_count = 0
  frame_count = 0
  fps_start = time.monotonic()
  fps = 0.0
  next_frame_time = 0.0  # initialized after warm-up

  try:
    # Warm-up ticks
    for _ in range(20):
      world.tick()
      tick_count += 1
    next_frame_time = time.monotonic()

    while running:
      world.tick()
      tick_count += 1

      if tick_count % TICKS_PER_FRAME != 0:
        continue

      road_rgb, wide_rgb = world.get_frame()

      # Choose display frame and send via VisionIPC
      if args.wide_road_only:
        if wide_rgb is None:
          continue
        display_rgb = wide_rgb
        yuv_wide = camerad.rgb_to_yuv(wide_rgb)
        camerad.cam_send_yuv_wide_road(yuv_wide)
      elif args.road_only:
        if road_rgb is None:
          continue
        display_rgb = road_rgb
        yuv_road = camerad.rgb_to_yuv(road_rgb)
        camerad.cam_send_yuv_road(yuv_road)
      else:
        if road_rgb is None:
          continue
        display_rgb = road_rgb
        yuv_road = camerad.rgb_to_yuv(road_rgb)
        camerad.cam_send_yuv_road(yuv_road)
        if wide_rgb is not None:
          yuv_wide = camerad.rgb_to_yuv(wide_rgb)
          camerad.cam_send_yuv_wide_road(yuv_wide)

      # Publish deviceState (modeld needs deviceType for DEVICE_CAMERAS lookup)
      publish_device_state(pm)

      # Publish carState (calibrationd needs vEgo)
      publish_car_state(pm, world.get_vehicle_speed())

      # Known pose mode: publish liveCalibration directly
      if not args.online_calib:
        publish_live_calibration(pm, rpyCalib, camera_height)

      # Non-blocking receive modelV2 and liveCalibration
      sm.update(0)

      # FPS tracking
      frame_count += 1
      now = time.monotonic()
      elapsed = now - fps_start
      if elapsed >= 2.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = now

      # Get current calibration from liveCalibration message
      if sm.seen['liveCalibration']:
        cur_rpyCalib = np.array(sm['liveCalibration'].rpyCalib)
        cur_height = float(sm['liveCalibration'].height[0]) if len(sm['liveCalibration'].height) > 0 else camera_height
        cur_valid_blocks = sm['liveCalibration'].validBlocks
        cur_cal_status = str(sm['liveCalibration'].calStatus)
        cur_cal_perc = int(sm['liveCalibration'].calPerc)
      else:
        cur_rpyCalib = rpyCalib
        cur_height = camera_height
        cur_valid_blocks = 0
        cur_cal_status = 'uncalibrated'
        cur_cal_perc = 0

      # Visualize (show last received model data, or None if never received)
      model_msg = sm['modelV2'] if sm.seen['modelV2'] else None
      ok = visualizer.draw(
        display_rgb, model_msg, vis_intrinsics,
        cur_rpyCalib, cur_height,
        world.get_vehicle_speed(), cur_cal_status,
        cur_valid_blocks, cur_cal_perc, fps)

      if not ok:
        break

      if args.max_frames > 0 and tick_count // TICKS_PER_FRAME >= args.max_frames:
        print(f"Reached max frames ({args.max_frames})")
        break

      if tick_count % 100 == 0:
        speed = world.get_vehicle_speed()
        modeld_status = 'connected' if sm.seen['modelV2'] else 'waiting...'
        calib_status = 'connected' if sm.seen['liveCalibration'] else 'waiting...'
        pitch_d, yaw_d = np.degrees(cur_rpyCalib[1]), np.degrees(cur_rpyCalib[2])
        print(f"[DASHCAM] frame={tick_count//TICKS_PER_FRAME} speed={speed:.1f}m/s "
              + f"pitch={pitch_d:.2f}\u00b0 yaw={yaw_d:.2f}\u00b0 fps={fps:.1f} modeld={modeld_status} "
              + f"calib={calib_status} calPerc={cur_cal_perc}% blocks={cur_valid_blocks}/{5} status={cur_cal_status}")

      # Frame rate limiter: sleep until next 50ms boundary for real-time playback
      if not args.fast:
        next_frame_time += FRAME_DT
        sleep_time = next_frame_time - time.monotonic()
        if sleep_time > 0:
          time.sleep(sleep_time)
        elif sleep_time < -FRAME_DT:
          next_frame_time = time.monotonic()

  except Exception as e:
    print(f"Error: {e}")
    raise
  finally:
    visualizer.close()
    # Terminate subprocesses first (before destroying Carla actors)
    for proc in [calibrationd_proc, modeld_proc]:
      if proc is not None:
        proc.terminate()
    for proc in [calibrationd_proc, modeld_proc]:
      if proc is not None:
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()
    # Note: skip world.close() explicit actor destroy - Carla cleans up on client disconnect.
    # Calling s.destroy() can trigger C++ std::runtime_error that bypasses Python exception handling.
    try:
      world.vehicle.set_autopilot(False)
    except Exception:
      pass
    print("Done")
    os._exit(0)


if __name__ == "__main__":
  main()
