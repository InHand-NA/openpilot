#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import threading
from multiprocessing import Process, Queue

from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType

# Ensure we can import from this folder despite hyphen in directory name
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
  sys.path.append(str(THIS_DIR))

from bridge.carla_bridge import CarlaBridge
from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread


def parse_args():
  parser = argparse.ArgumentParser(description='Bridge between CARLA simulator and openpilot (carla 0.9.15).')
  parser.add_argument('--host', type=str, default='127.0.0.1')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--town', type=str, default='Town03')
  parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=0)
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  return parser.parse_args()


def main():
  args = parse_args()

  q: Queue = Queue()

  bridge = CarlaBridge(args.dual_camera, args.high_quality,
                       host=args.host, port=args.port,
                       town=args.town,
                       num_selected_spawn_point=args.num_selected_spawn_point)

  # start input thread (keyboard by default). Joystick can be run as separate process.
  kb_thread = threading.Thread(target=keyboard_poll_thread, args=(q,), daemon=True)
  kb_thread.start()

  p = bridge.run(q)

  try:
    p.join()
  except KeyboardInterrupt:
    bridge.shutdown()
    q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, 'keyboard interrupt'))
    p.join()


if __name__ == '__main__':
  main()
