#!/usr/bin/env python3
import argparse

from typing import Any
from multiprocessing import Queue

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
from openpilot.tools.sim.bridge.carla.carla_bridge import CarlaBridge

def create_bridge(simulator_type, dual_camera, high_quality, carla_autopilot=False, carla_autopilot_speed=35.0, perfect_cam=False):
  queue: Any = Queue()

  simulator_bridge: SimulatorBridge
  if simulator_type == 'metadrive':
    simulator_bridge = MetaDriveBridge(dual_camera, high_quality)
  elif simulator_type == 'carla':
    simulator_bridge = CarlaBridge(dual_camera, high_quality, carla_autopilot=carla_autopilot,
                                   carla_autopilot_speed=carla_autopilot_speed,
                                   perfect_cam=perfect_cam)
  else:
    raise ValueError(f"Unknown simulator type: {simulator_type}")

  simulator_process = simulator_bridge.run(queue)

  return queue, simulator_process, simulator_bridge

def main():
  _, simulator_process, _ = create_bridge('carla', True, False)
  simulator_process.join()

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--simulator', dest='simulator', type=str, default='carla')
  parser.add_argument('--carla_autopilot', action='store_true', help='Use Carla autopilot instead of openpilot control')
  parser.add_argument('--carla_autopilot_speed', type=float, default=35.0, help='Carla autopilot target speed in MPH (default: 35)')
  parser.add_argument('--perfect_cam', action='store_true',
                      help='Use standard camera pose (no pitch/yaw offset) and skip calibration')

  return parser.parse_args(add_args)

if __name__ == "__main__":
  args = parse_args()

  queue, simulator_process, simulator_bridge = create_bridge(args.simulator,
                                                             args.dual_camera, args.high_quality,
                                                             carla_autopilot=args.carla_autopilot,
                                                             carla_autopilot_speed=args.carla_autopilot_speed,
                                                             perfect_cam=args.perfect_cam)

  if args.joystick:
    # start input poll for joystick
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(queue)
  else:
    # start input poll for keyboard
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(queue)

  simulator_bridge.shutdown()

  simulator_process.join()
