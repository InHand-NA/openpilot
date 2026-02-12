#!/usr/bin/env python3
"""Standalone dashcam with openpilot vision network.

Usage:
  # Start Carla server first, then:
  python tools/dashcam/run.py
  python tools/dashcam/run.py --perfect-cam
  python tools/dashcam/run.py --save-video output.mp4
  python tools/dashcam/run.py --camera-pitch 5 --camera-yaw 3 --online-calib
"""

import argparse
import signal
import sys
import time

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS

TICKS_PER_FRAME = 5  # match openpilot sim bridge


def main():
  parser = argparse.ArgumentParser(description='Standalone dashcam with openpilot vision')
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
                      help='Use online calibration instead of known pose')
  parser.add_argument('--num-npc', type=int, default=20)
  parser.add_argument('--high-quality', action='store_true')
  parser.add_argument('--no-display', action='store_true')
  parser.add_argument('--save-video', type=str, default='',
                      help='Save visualization to mp4 file')
  parser.add_argument('--max-frames', type=int, default=0,
                      help='Stop after N frames (0 = unlimited)')
  args = parser.parse_args()

  # Camera pose
  if args.perfect_cam:
    pitch_deg, yaw_deg = 0.0, 0.0
  else:
    pitch_deg, yaw_deg = args.camera_pitch, args.camera_yaw

  print(f"Camera pose: pitch={pitch_deg}° yaw={yaw_deg}° height={args.camera_height}m")
  print(f"Calibration mode: {'online' if args.online_calib else 'known pose'}")

  # Initialize Carla world
  print("Connecting to Carla...")
  from openpilot.tools.dashcam.carla_world import DashcamCarlaWorld
  world = DashcamCarlaWorld(
    host=args.host, port=args.port, town=args.town,
    spawn_point=args.spawn_point,
    camera_pitch_deg=pitch_deg, camera_yaw_deg=yaw_deg,
    camera_height=args.camera_height,
    high_quality=args.high_quality, num_npc=args.num_npc)

  # Initialize vision model
  print("Initializing vision model...")
  from openpilot.tools.dashcam.vision_model import VisionModel
  model = VisionModel()

  # Initialize calibrator
  from openpilot.tools.dashcam.calibrator import KnownPoseCalibrator, OnlineCalibrator
  if args.online_calib:
    calibrator = OnlineCalibrator(
      pitch_deg_init=pitch_deg, yaw_deg_init=yaw_deg,
      height_init=args.camera_height)
  else:
    calibrator = KnownPoseCalibrator(
      pitch_deg=pitch_deg, yaw_deg=yaw_deg,
      height=args.camera_height)

  # Camera intrinsics
  dc = DEVICE_CAMERAS[("pc", "unknown")]

  # Initialize visualizer
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
  print("Starting main loop...")
  tick_count = 0
  frame_count = 0
  fps_start = time.monotonic()
  fps = 0.0
  last_output = None

  try:
    # Warm-up ticks
    for _ in range(20):
      world.tick()
      tick_count += 1

    while running:
      world.tick()
      tick_count += 1

      if tick_count % TICKS_PER_FRAME != 0:
        continue

      road_rgb, wide_rgb = world.get_frame()
      if road_rgb is None:
        continue

      # Preprocess + inference
      t0 = time.monotonic()
      vision_inputs = model.preprocess(road_rgb, wide_rgb, dc, calibrator.rpyCalib)
      output = model.run(vision_inputs)
      t1 = time.monotonic()
      last_output = output

      # Update calibrator
      calibrator.update(output, world.get_vehicle_speed())

      # FPS tracking
      frame_count += 1
      now = time.monotonic()
      elapsed = now - fps_start
      if elapsed >= 2.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = now

      # Visualize
      ok = visualizer.draw(
        road_rgb, output, dc.fcam.intrinsics,
        calibrator.rpyCalib, calibrator.height,
        world.get_vehicle_speed(), calibrator.cal_status,
        calibrator.valid_blocks, fps)

      if not ok:
        break

      if args.max_frames > 0 and tick_count // TICKS_PER_FRAME >= args.max_frames:
        print(f"Reached max frames ({args.max_frames})")
        break

      if tick_count % 100 == 0:
        speed = world.get_vehicle_speed()
        rpy = calibrator.rpyCalib
        print(f"[DASHCAM] frame={tick_count//TICKS_PER_FRAME} speed={speed:.1f}m/s "
              f"pitch={np.degrees(rpy[1]):.2f}° yaw={np.degrees(rpy[2]):.2f}° "
              f"model_time={t1-t0:.3f}s fps={fps:.1f}")

  except Exception as e:
    print(f"Error: {e}")
    raise
  finally:
    # Ensure cleanup runs even without atexit
    visualizer.close()
    world.close()
    print("Done")


if __name__ == "__main__":
  main()
