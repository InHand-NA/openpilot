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
  parser.add_argument('--random-spawn', action='store_true', help='Spawn ego at a random waypoint (overrides --spawn-point)')
  parser.add_argument('--camera-pitch', type=float, default=5.0, help='Camera pitch in degrees')
  parser.add_argument('--camera-yaw', type=float, default=3.0, help='Camera yaw in degrees')
  parser.add_argument('--camera-height', type=float, default=1.13, help='Camera height in meters')
  parser.add_argument('--perfect-cam', action='store_true', help='Use pitch=0, yaw=0 (ideal mounting)')
  parser.add_argument('--online-calib', action='store_true', help='Use online calibration (calibrationd subprocess)')
  parser.add_argument('--num-npc', type=int, default=20)
  parser.add_argument('--high-quality', action='store_true')
  parser.add_argument('--no-display', action='store_true')
  parser.add_argument('--save-video', type=str, default='', help='Save visualization to mp4 file')
  parser.add_argument('--max-frames', type=int, default=0, help='Stop after N frames (0 = unlimited)')
  parser.add_argument('--fast', action='store_true', help='Run as fast as possible, bypass frame rate limiter')
  parser.add_argument('--wide-road-only', action='store_true', help='Single wide camera mode (modeld uses ecam intrinsics for both inputs)')
  parser.add_argument('--road-only', action='store_true', help='Single narrow camera mode (modeld uses fcam intrinsics for main input)')
  parser.add_argument('--height-comp', action='store_true', help='Enable lane line height compensation, do not use this functionality')
  parser.add_argument('--eval-lanes', action='store_true', help='Enable lane line ground truth evaluation')
  parser.add_argument('--eval-interval', type=int, default=1, help='GT evaluation interval in frames (default: every frame)')
  parser.add_argument('--record', type=str, default='', help='Enable training data recording, save to specified directory')
  parser.add_argument('--record-skip', type=int, default=1, help='Save every N-th frame when recording (default: 1)')
  parser.add_argument(
    '--speed-range', type=float, nargs=2, default=[20.0, 140.0], metavar=('MIN', 'MAX'), help='Ego target speed range in km/h (default: 20 140)'
  )
  parser.add_argument('--record-only', action='store_true', help='Record mode: disable modeld and visualization, only collect GT data')
  parser.add_argument('--custom-model', type=str, default='', help='Path to custom ONNX model (bypasses modeld, uses onnxruntime)')
  parser.add_argument('--record-modeld', type=str, default='', help='Record dual-camera data with modeld labels to specified directory')
  parser.add_argument('--custom-modeld', type=str, default='', help='Custom model .pkl or .onnx (ONNX auto-compiles to tinygrad pkl)')
  args = parser.parse_args()

  if args.wide_road_only and args.road_only:
    parser.error('--wide-road-only and --road-only are mutually exclusive')

  # Record mode: auto-enable road-only and fast for efficient data collection
  recording = bool(args.record)
  record_only = args.record_only
  if recording:
    if not args.road_only and not args.wide_road_only:
      args.road_only = True
    args.fast = True
    print(f"[RECORD] Recording enabled -> {args.record}")
    if record_only:
      args.no_display = True
      print("[RECORD] Record-only mode: modeld and visualization disabled")

  custom_model_mode = bool(args.custom_model)
  if custom_model_mode:
    if not args.road_only and not args.wide_road_only:
      args.road_only = True
    print(f"[CustomModel] Using custom ONNX model: {args.custom_model}")

  # Dual-camera recording with modeld labels
  record_modeld = bool(args.record_modeld)
  if record_modeld:
    args.fast = True
    # Don't force road_only — dual-camera needs both streams
    print(f"[RECORD-MODELD] Dual-camera recording with modeld labels -> {args.record_modeld}")

  # Custom modeld mode (tinygrad pkl subprocess, accepts .pkl or .onnx)
  custom_modeld_mode = bool(args.custom_modeld)
  if custom_modeld_mode:
    # Don't force road_only — custom_modeld handles dual streams
    print(f"[CustomModeld] Using custom model: {args.custom_modeld}")

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

  # 1. Initialize Params and subprocesses (skip in record-only mode)
  camerad = None
  modeld_proc = None
  calibrationd_proc = None
  pm = None
  sm = None

  # Standard modeld mode
  use_standard_modeld = not record_only and not custom_model_mode and not custom_modeld_mode

  if use_standard_modeld or record_modeld:
    params = Params()

    from opendbc.car.car_helpers import get_demo_car_params

    CP = get_demo_car_params()
    params.put("CarParams", CP.to_bytes())

    calib_msg = messaging.new_message('liveCalibration')
    if args.online_calib:
      calib_msg.liveCalibration.validBlocks = 0
      calib_msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    else:
      calib_msg.liveCalibration.validBlocks = 20
      calib_msg.liveCalibration.rpyCalib = rpyCalib.tolist()
    params.put("CalibrationParams", calib_msg.to_bytes())

    print("Creating VisionIPC server...")
    camerad = DashcamCamerad(wide_road_only=args.wide_road_only, road_only=args.road_only)

    modeld_env = {**os.environ, 'DEV': 'CUDA', 'PYOPENCL_CTX': ''}
    print(f"Starting modeld subprocess (DEV={modeld_env['DEV']})...")
    modeld_proc = subprocess.Popen([sys.executable, '-m', 'selfdrive.modeld.modeld'], env=modeld_env)

    if args.online_calib:
      print("Starting calibrationd subprocess...")
      calibrationd_proc = subprocess.Popen([sys.executable, '-m', 'openpilot.tools.dashcam.calibrationd'], env={**os.environ})

    sub_topics = ['modelV2', 'liveCalibration']
    if record_modeld:
      sub_topics.append('cameraOdometry')
    pub_services = ['carState', 'deviceState']
    if not args.online_calib:
      pub_services.append('liveCalibration')
    pm = messaging.PubMaster(pub_services)
    sm = messaging.SubMaster(sub_topics)

  # Custom modeld mode: launch custom_modeld subprocess
  if custom_modeld_mode:
    params = Params()

    from opendbc.car.car_helpers import get_demo_car_params

    CP = get_demo_car_params()
    params.put("CarParams", CP.to_bytes())

    calib_msg = messaging.new_message('liveCalibration')
    if args.online_calib:
      calib_msg.liveCalibration.validBlocks = 0
      calib_msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    else:
      calib_msg.liveCalibration.validBlocks = 20
      calib_msg.liveCalibration.rpyCalib = rpyCalib.tolist()
    params.put("CalibrationParams", calib_msg.to_bytes())

    print("Creating VisionIPC server...")
    camerad = DashcamCamerad(wide_road_only=args.wide_road_only, road_only=args.road_only)

    model_path = args.custom_modeld

    if model_path.endswith('.onnx'):
      # Auto-compile ONNX → tinygrad pkl + metadata pkl
      base = os.path.splitext(model_path)[0]
      dev = os.environ.get('DEV', 'CUDA').lower()
      pkl_path = f"{base}_tinygrad_{dev}.pkl"
      metadata_path = f"{base}_metadata.pkl"

      needs_compile = not os.path.exists(pkl_path) or os.path.getmtime(model_path) > os.path.getmtime(pkl_path)
      if needs_compile:
        print(f"[CustomModeld] Compiling ONNX → tinygrad pkl...")
        from openpilot.tools.dashcam.train.compile_tinygrad import compile_model, generate_metadata
        generate_metadata(model_path, metadata_path)
        compile_model(model_path, pkl_path)
        print(f"[CustomModeld] Compilation done: {pkl_path}")
      else:
        print(f"[CustomModeld] Using cached pkl: {pkl_path}")
    else:
      # Existing .pkl path logic: find metadata pkl alongside the model pkl
      pkl_path = model_path
      pkl_base = os.path.splitext(pkl_path)[0]
      # Try: same_dir/*_metadata.pkl or strip _tinygrad_xxx suffix
      metadata_candidates = [
        pkl_base.rsplit('_tinygrad', 1)[0] + '_metadata.pkl',
        pkl_base + '_metadata.pkl',
      ]
      metadata_path = ''
      for c in metadata_candidates:
        if os.path.exists(c):
          metadata_path = c
          break
      if not metadata_path:
        raise FileNotFoundError(f"Cannot find metadata pkl for {pkl_path}. Tried: {metadata_candidates}")

    modeld_env = {
      **os.environ,
      'DEV': 'CUDA',
      'PYOPENCL_CTX': '',
      'CUSTOM_MODEL_PKL': os.path.abspath(pkl_path),
      'CUSTOM_MODEL_METADATA': os.path.abspath(metadata_path),
    }
    print(f"Starting custom_modeld subprocess (pkl={pkl_path}, metadata={metadata_path})...")
    modeld_proc = subprocess.Popen([sys.executable, '-m', 'openpilot.tools.dashcam.custom_modeld'], env=modeld_env)

    if args.online_calib:
      print("Starting calibrationd subprocess...")
      calibrationd_proc = subprocess.Popen([sys.executable, '-m', 'openpilot.tools.dashcam.calibrationd'], env={**os.environ})

    pub_services = ['carState', 'deviceState']
    if not args.online_calib:
      pub_services.append('liveCalibration')
    pm = messaging.PubMaster(pub_services)
    sm = messaging.SubMaster(['modelV2', 'liveCalibration'])

  # Custom model inference (in-process ONNX, bypasses modeld)
  inference = None
  if custom_model_mode:
    from openpilot.tools.dashcam.infer import CustomModelInference

    dc = DEVICE_CAMERAS[("pc", "unknown")]
    inference = CustomModelInference(args.custom_model, dc.fcam.intrinsics)

  # 6. Connect to Carla
  print("Connecting to Carla...")
  from openpilot.tools.dashcam.carla_world import DashcamCarlaWorld

  world = DashcamCarlaWorld(
    host=args.host,
    port=args.port,
    town=args.town,
    spawn_point=args.spawn_point,
    random_spawn=args.random_spawn,
    camera_pitch_deg=pitch_deg,
    camera_yaw_deg=yaw_deg,
    camera_height=camera_height,
    high_quality=args.high_quality,
    num_npc=args.num_npc,
    wide_road_only=args.wide_road_only,
    road_only=args.road_only,
    speed_range=tuple(args.speed_range),
  )

  # Lane GT evaluation
  gt_extractor = None
  evaluator = None
  if args.eval_lanes:
    from openpilot.tools.dashcam.lane_evaluator import LaneEvaluator
    from openpilot.tools.dashcam.lane_ground_truth import LaneGroundTruth

    gt_extractor = LaneGroundTruth(world.get_map(), camera_offset_x=0.8, camera_height=camera_height)
    evaluator = LaneEvaluator()
    print(f"Lane GT evaluation enabled (interval={args.eval_interval})")

  # Dual-camera recording with modeld labels
  dual_recorder = None
  label_extractor = None
  if record_modeld:
    from openpilot.tools.dashcam.dual_data_recorder import DualCameraDataRecorder
    from openpilot.tools.dashcam.modeld_label_extractor import ModeldLabelExtractor

    label_extractor = ModeldLabelExtractor()
    clip_metadata = world.get_clip_metadata(args.town)
    dual_recorder = DualCameraDataRecorder(
      output_dir=args.record_modeld, town=args.town, camera_height=camera_height, label_source='modeld', skip_frames=args.record_skip, metadata=clip_metadata
    )
    print("[RECORD-MODELD] Label extractor + dual recorder initialized")

  # Training data recording
  recorder = None
  lane_gt_extractor = None
  lead_gt_extractor = None
  pose_gt_extractor = None
  if recording:
    from openpilot.tools.dashcam.data_recorder import DataRecorder
    from openpilot.tools.dashcam.lane_ground_truth import LaneGroundTruth
    from openpilot.tools.dashcam.lead_ground_truth import LeadGroundTruth
    from openpilot.tools.dashcam.pose_ground_truth import PoseGroundTruth

    lane_gt_extractor = LaneGroundTruth(world.get_map(), camera_offset_x=0.8, camera_height=camera_height)
    lead_gt_extractor = LeadGroundTruth(world.get_world(), world.get_vehicle(), camera_offset_x=0.8, camera_height=camera_height)
    pose_gt_extractor = PoseGroundTruth(camera_offset_x=0.8, camera_height=camera_height)
    clip_metadata = world.get_clip_metadata(args.town)
    recorder = DataRecorder(
      output_dir=args.record,
      town=args.town,
      camera_height=camera_height,
      camera_pitch=np.deg2rad(pitch_deg),
      camera_yaw=np.deg2rad(yaw_deg),
      skip_frames=args.record_skip,
      metadata=clip_metadata,
    )
    print("[RECORD] GT extractors initialized (lane + lead + pose)")

  # Camera intrinsics and visualizer (skip in record-only mode)
  vis_intrinsics = None
  visualizer = None
  if not record_only:
    # wide-road-only: modeld uses ecam.intrinsics for both transforms, so visualization must match
    # road-only / dual: display the narrow road camera, use fcam.intrinsics
    dc = DEVICE_CAMERAS[("pc", "unknown")]
    vis_intrinsics = dc.ecam.intrinsics if args.wide_road_only else dc.fcam.intrinsics

    from openpilot.tools.dashcam.visualizer import Visualizer

    visualizer = Visualizer(
      save_video_path=args.save_video,
      no_display=args.no_display,
      source_fps=20.0,
      actual_height=camera_height if args.height_comp else 0.0,
      show_bev=custom_model_mode or custom_modeld_mode,
    )

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
    # Warm-up ticks (not counted towards max_frames)
    for _ in range(20):
      world.tick()
    tick_count = 0
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
      elif args.road_only:
        if road_rgb is None:
          continue
        display_rgb = road_rgb
      else:
        if road_rgb is None:
          continue
        display_rgb = road_rgb

      # Send frames to modeld via VisionIPC (skip in record-only and custom-model ONNX modes)
      if not record_only and not custom_model_mode:
        if args.wide_road_only:
          yuv_wide = camerad.rgb_to_yuv(wide_rgb)
          camerad.cam_send_yuv_wide_road(yuv_wide)
        elif args.road_only:
          yuv_road = camerad.rgb_to_yuv(road_rgb)
          camerad.cam_send_yuv_road(yuv_road)
        else:
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

      # Custom model inference (in-process ONNX)
      custom_model_msg = None
      if custom_model_mode:
        result = inference.process_frame(display_rgb, rpyCalib)
        custom_model_msg = result.modelV2 if result is not None else None

      # Training data recording
      if recorder is not None:
        veh_transform = world.get_vehicle_transform()
        v_ego = world.get_vehicle_speed()

        rec_lane_gt = lane_gt_extractor.get_lane_lines(veh_transform)
        rec_road_edges_gt = lane_gt_extractor.get_road_edges(veh_transform)
        rec_road_edges_gt = lane_gt_extractor.filter_road_edges(rec_lane_gt, rec_road_edges_gt)
        rec_lead_gt = lead_gt_extractor.get_lead_vehicles(veh_transform, v_ego, road_edges=rec_road_edges_gt)
        rec_pose, rec_road_transform, rec_wide_from_device_euler = pose_gt_extractor.update(veh_transform)

        # Per-frame rpyCalib: camera mounting angles + vehicle tilt
        veh_pitch_rad = np.deg2rad(veh_transform.rotation.pitch)
        veh_roll_rad = np.deg2rad(veh_transform.rotation.roll)
        frame_rpyCalib = np.array(
          [
            veh_roll_rad,
            -(np.deg2rad(pitch_deg) + veh_pitch_rad),
            -np.deg2rad(yaw_deg),
          ],
          dtype=np.float32,
        )

        world_pose = np.array(
          [
            veh_transform.location.x,
            veh_transform.location.y,
            veh_transform.location.z,
            veh_transform.rotation.roll,
            veh_transform.rotation.pitch,
            veh_transform.rotation.yaw,
          ],
          dtype=np.float32,
        )

        recorder.record_with_vego(
          display_rgb,
          rec_lane_gt,
          rec_lead_gt,
          rec_pose,
          rec_road_transform,
          v_ego,
          road_edges_gt=rec_road_edges_gt,
          rpyCalib=frame_rpyCalib,
          world_pose=world_pose,
          wide_from_device_euler=rec_wide_from_device_euler,
        )

      # Dual-camera recording with modeld labels
      if dual_recorder is not None and sm is not None:
        if sm.seen['modelV2']:
          cam_odom = sm['cameraOdometry'] if sm.seen.get('cameraOdometry', False) else None
          labels = label_extractor.extract(sm['modelV2'], cam_odom)
          if labels is not None and road_rgb is not None and wide_rgb is not None:
            veh_transform = world.get_vehicle_transform()
            veh_pitch_rad = np.deg2rad(veh_transform.rotation.pitch)
            veh_roll_rad = np.deg2rad(veh_transform.rotation.roll)
            frame_rpyCalib_dual = np.array(
              [
                veh_roll_rad,
                -(np.deg2rad(pitch_deg) + veh_pitch_rad),
                -np.deg2rad(yaw_deg),
              ],
              dtype=np.float32,
            )
            world_pose_dual = np.array(
              [
                veh_transform.location.x,
                veh_transform.location.y,
                veh_transform.location.z,
                veh_transform.rotation.roll,
                veh_transform.rotation.pitch,
                veh_transform.rotation.yaw,
              ],
              dtype=np.float32,
            )
            dual_recorder.record(road_rgb, wide_rgb, labels, rpyCalib=frame_rpyCalib_dual, v_ego=world.get_vehicle_speed(), world_pose=world_pose_dual)

      # Lane GT evaluation (non-recording path)
      gt_lines = None
      gt_probs = None
      eval_metrics = None
      eval_frame_count = tick_count // TICKS_PER_FRAME
      if gt_extractor is not None and eval_frame_count % args.eval_interval == 0:
        gt_lines, gt_probs = gt_extractor.get_lane_lines(world.get_vehicle_transform())
        if gt_lines is not None and sm is not None:
          model_msg_for_eval = sm['modelV2'] if sm.seen['modelV2'] else None
          if model_msg_for_eval is not None and evaluator is not None:
            model_lane_lines = [np.array([ll.x, ll.y, ll.z], dtype=np.float32).T for ll in model_msg_for_eval.laneLines]
            model_probs = list(model_msg_for_eval.laneLineProbs)
            eval_metrics = evaluator.evaluate(model_lane_lines, model_probs, gt_lines, gt_probs)

      # FPS tracking
      frame_count += 1
      now = time.monotonic()
      elapsed = now - fps_start
      if elapsed >= 2.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = now

      if not record_only:
        # Get current calibration and model output
        if custom_model_mode:
          cur_rpyCalib = rpyCalib
          cur_height = camera_height
          cur_valid_blocks = 20
          cur_cal_status = 'calibrated'
          cur_cal_perc = 100
          model_msg = custom_model_msg
        else:
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
          model_msg = sm['modelV2'] if sm.seen['modelV2'] else None

        ok = visualizer.draw(
          display_rgb,
          model_msg,
          vis_intrinsics,
          cur_rpyCalib,
          cur_height,
          world.get_vehicle_speed(),
          cur_cal_status,
          cur_valid_blocks,
          cur_cal_perc,
          fps,
          gt_lines=gt_lines,
          gt_probs=gt_probs,
          eval_metrics=eval_metrics,
        )

        if not ok:
          break

      if args.max_frames > 0 and tick_count // TICKS_PER_FRAME >= args.max_frames:
        print(f"Reached max frames ({args.max_frames})")
        break

      if tick_count % 100 == 0:
        speed = world.get_vehicle_speed()
        if record_only:
          saved = recorder.saved_count if recorder else 0
          print(f"[RECORD] frame={tick_count // TICKS_PER_FRAME} speed={speed:.1f}m/s " + f"fps={fps:.1f} saved={saved}")
        elif custom_model_mode:
          has_output = custom_model_msg is not None
          print(
            f"[CustomModel] frame={tick_count // TICKS_PER_FRAME} speed={speed:.1f}m/s " + f"fps={fps:.1f} output={'active' if has_output else 'buffering'}"
          )
        elif custom_modeld_mode:
          modeld_status = 'connected' if sm.seen['modelV2'] else 'waiting...'
          dual_saved = dual_recorder.saved_count if dual_recorder else '-'
          print(f"[CustomModeld] frame={tick_count // TICKS_PER_FRAME} speed={speed:.1f}m/s " + f"fps={fps:.1f} modeld={modeld_status} saved={dual_saved}")
        elif record_modeld:
          modeld_status = 'connected' if sm.seen['modelV2'] else 'waiting...'
          dual_saved = dual_recorder.saved_count if dual_recorder else 0
          print(f"[RECORD-MODELD] frame={tick_count // TICKS_PER_FRAME} speed={speed:.1f}m/s " + f"fps={fps:.1f} modeld={modeld_status} saved={dual_saved}")
        else:
          modeld_status = 'connected' if sm.seen['modelV2'] else 'waiting...'
          calib_status = 'connected' if sm.seen['liveCalibration'] else 'waiting...'
          pitch_d, yaw_d = np.degrees(cur_rpyCalib[1]), np.degrees(cur_rpyCalib[2])
          print(
            f"[DASHCAM] frame={tick_count // TICKS_PER_FRAME} speed={speed:.1f}m/s "
            + f"pitch={pitch_d:.2f}\u00b0 yaw={yaw_d:.2f}\u00b0 fps={fps:.1f} modeld={modeld_status} "
            + f"calib={calib_status} calPerc={cur_cal_perc}% blocks={cur_valid_blocks}/{5} status={cur_cal_status}"
          )

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
    if recorder is not None:
      recorder.close()
    if dual_recorder is not None:
      dual_recorder.close()
    if evaluator is not None and evaluator.history:
      summary = evaluator.get_summary()
      print("\n=== Lane Evaluation Summary ===")
      for k, v in sorted(summary.items()):
        print(f"  {k}: {v:.4f}")
      print(f"  total_frames: {len(evaluator.history)}")
    if not record_only:
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
