from multiprocessing import Queue

from openpilot.tools.sim.bridge.common import SimulatorBridge
from .carla_world import CarlaWorld


class CarlaBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, host: str, port: int, town: str, num_selected_spawn_point: int):
    super().__init__(dual_camera, high_quality)
    self.host = host
    self.port = port
    self.town = town
    self.num_selected_spawn_point = num_selected_spawn_point

  def spawn_world(self, q: Queue) -> CarlaWorld:
    import carla

    client = carla.Client(self.host, self.port)
    client.set_timeout(5)

    return CarlaWorld(q, client, high_quality=self.high_quality, dual_camera=self.dual_camera,
                      num_selected_spawn_point=self.num_selected_spawn_point, town=self.town)

