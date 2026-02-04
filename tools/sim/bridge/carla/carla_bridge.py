import math
import os

from multiprocessing import Queue
from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.carla.carla_world import CarlaWorld


class CarlaBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False):
    super().__init__(dual_camera, high_quality)
    self.host = '127.0.0.1' # arguments.host
    self.port = 2000 # arguments.port
    self.town = 'Town04_Opt' # arguments.town
    self.num_selected_spawn_point = 16 # arguments.num_selected_spawn_point

  def spawn_world(self, q: Queue):
    import carla

    client = carla.Client(self.host, self.port)
    # 允许通过环境变量调整 CARLA RPC 超时（单位：秒），默认 15s
    # 示例：export CARLA_CLIENT_TIMEOUT_S=20
    try:
      rpc_timeout_s = float(os.environ.get("CARLA_CLIENT_TIMEOUT_S", "60"))
    except ValueError:
      rpc_timeout_s = 60.0
    #client.set_timeout(rpc_timeout_s)
    client.set_timeout(10.0)

    return CarlaWorld(client, high_quality=self.high_quality, dual_camera=self.dual_camera,
                      num_selected_spawn_point=self.num_selected_spawn_point, town=self.town)
