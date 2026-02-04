import numpy as np

from openpilot.common.params import Params
from openpilot.tools.sim.lib.common import SimulatorState, vec3
from openpilot.tools.sim.bridge.common import World
from openpilot.tools.sim.lib.camerad import W, H


class CarlaWorld(World):
  def __init__(self, client, high_quality, dual_camera, num_selected_spawn_point, town):
    super().__init__(dual_camera)
    import carla

    low_quality_layers = carla.MapLayer(carla.MapLayer.Ground | carla.MapLayer.Walls | carla.MapLayer.Decals)

    layers = carla.MapLayer.All if high_quality else low_quality_layers

    world = client.load_world(town, map_layers=layers)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    self.world = world
    world_map = world.get_map()

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = world_map.get_spawn_points()
    assert len(spawn_points) > num_selected_spawn_point, \
      f'''No spawn point {num_selected_spawn_point}, try a value between 0 and {len(spawn_points)} for this town.'''
    self.spawn_point = spawn_points[num_selected_spawn_point]
    self.vehicle = world.spawn_actor(vehicle_bp, self.spawn_point)

    physics_control = self.vehicle.get_physics_control()
    physics_control.mass = 2326
    physics_control.torque_curve = [carla.Vector2D(20.0, 500.0), carla.Vector2D(5000.0, 500.0)]
    physics_control.gear_switch_time = 0.0
    self.vehicle.apply_physics_control(physics_control)

    self.vc: carla.VehicleControl = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)
    self.max_steer_angle: float = self.vehicle.get_physics_control().wheels[0].max_steer_angle
    self.params = Params()

    self.steer_ratio = 15

    self.carla_objects = []

    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      blueprint.set_attribute('sensor_tick', str(1/20))
      if not high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=self.vehicle)
      camera.listen(callback)
      return camera

    self.road_camera = create_camera(fov=40, callback=self.cam_callback_road)
    if dual_camera:
      self.road_wide_camera = create_camera(fov=120, callback=self.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts
    else:
      self.road_wide_camera = None

    # re-enable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    self.imu = world.spawn_actor(imu_bp, transform, attach_to=self.vehicle)

    gps_bp = blueprint_library.find('sensor.other.gnss')
    self.gps = world.spawn_actor(gps_bp, transform, attach_to=self.vehicle)
    self.params.put_bool("UbloxAvailable", True)

    self.carla_objects = [self.imu, self.gps, self.road_camera, self.road_wide_camera, self.vehicle]

  def close(self):
    for s in self.carla_objects:
      if s is not None:
        try:
          s.destroy()
        except Exception as e:
          print("Failed to destroy carla object", e)

  def carla_image_to_rgb(self, image):
    rgb = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    rgb = np.reshape(rgb, (H, W, 4))
    return np.ascontiguousarray(rgb[:, :, [0, 1, 2]])

  def cam_callback_road(self, image):
    with self.image_lock:
      self.road_image = self.carla_image_to_rgb(image)

  def cam_callback_wide_road(self, image):
    with self.image_lock:
      self.wide_road_image = self.carla_image_to_rgb(image)

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    self.vc.throttle = throttle_out

    steer_carla = steer_angle * -1 / (self.max_steer_angle * self.steer_ratio)
    steer_carla = np.clip(steer_carla, -1, 1)

    self.vc.steer = steer_carla
    self.vc.brake = brake_out
    self.vehicle.apply_control(self.vc)

  def read_state(self):
    # DO we need this method?
    pass

  def read_sensors(self, simulator_state: SimulatorState):
    simulator_state.imu.bearing = self.imu.get_transform().rotation.yaw

    simulator_state.imu.accelerometer = vec3(
      self.imu.get_acceleration().x,
      self.imu.get_acceleration().y,
      self.imu.get_acceleration().z
    )

    simulator_state.imu.gyroscope = vec3(
      self.imu.get_angular_velocity().x,
      self.imu.get_angular_velocity().y,
      self.imu.get_angular_velocity().z
    )

    simulator_state.gps.from_xy([self.vehicle.get_location().x, self.vehicle.get_location().y])

    simulator_state.velocity = self.vehicle.get_velocity()
    simulator_state.valid = True
    simulator_state.steering_angle = self.vc.steer * self.max_steer_angle

  def read_cameras(self):
    pass # cameras are read within a callback for carla

  def tick(self):
    # 允许通过环境变量调整同步 tick 的超时（毫秒）。
    # CARLA 同步模式下默认等待 5s（底层异常提示），
    # 在本地加载地图/着色器首次较慢时容易触发超时。
    # 示例：export CARLA_TICK_TIMEOUT_MS=20000  # 20s
    import os
    timeout_ms_env = os.environ.get("CARLA_TICK_TIMEOUT_MS")
    if timeout_ms_env is not None:
      try:
        timeout_ms = int(timeout_ms_env)
        self.world.tick(timeout_ms=timeout_ms)
        return
      except Exception:
        # 回退到默认行为
        pass
    self.world.tick()

  def reset(self):
    import carla
    self.vehicle.set_transform(self.spawn_point)
    self.vehicle.set_target_velocity(carla.Vector3D())


def test_carla_world(host: str = None, port: int | None = None, timeout: float | None = None, spawn_test: bool = False) -> int:
  """
  连接并自检 CARLA Server 的核心通信能力，不依赖 openpilot 运行时。

  检查项（按顺序）：
  - 创建 `carla.Client` 并设置超时
  - 读取服务端/客户端版本，拉取当前 `World`
  - 读取 `Map` 名称与 `Settings`
  - `wait_for_tick` 获取一次 `WorldSnapshot`（验证时钟/消息通道）
  - 读取 `Actors` 列表与 `Spectator` 位姿
  - （可选）临时修改一次世界渲染帧率设置并回滚；或轻量 Spawn/Destroy 演示

  返回 0 表示成功，非 0 表示失败。
  """
  import os
  import sys
  import contextlib

  # 延迟导入，避免对本文件其他依赖的影响
  try:
    import carla  # type: ignore
  except Exception as e:  # pragma: no cover - 依赖环境
    print("[ERROR] 未找到 carla Python 模块，请确认已安装 CARLA wheel 并设置 PYTHONPATH。\n", e)
    return 2

  host = host or os.environ.get("CARLA_HOST", "127.0.0.1")
  port = int(port if port is not None else os.environ.get("CARLA_PORT", 2000))
  timeout = float(timeout if timeout is not None else os.environ.get("CARLA_TIMEOUT", 5.0))

  print(f"[INFO] 尝试连接 CARLA Server {host}:{port}，超时 {timeout:.1f}s …")

  try:
    client = carla.Client(host, port)
    client.set_timeout(timeout)
  except Exception as e:
    print("[ERROR] 创建 carla.Client 失败:", e)
    return 3

  try:
    server_ver = client.get_server_version()
    client_ver = getattr(carla, "__version__", "unknown")
    print(f"[OK] 版本: server={server_ver}, client={client_ver}")

    world = client.get_world()
    if world is None:
      print("[ERROR] 获取 World 失败: 返回 None")
      return 4

    world_map = world.get_map()
    map_name = getattr(world_map, "name", "<unknown>")
    print(f"[OK] 地图: {map_name}")

    settings = world.get_settings()
    print(
      f"[OK] 设置: synchronous={settings.synchronous_mode}, fixed_delta={settings.fixed_delta_seconds}"
    )

    # 验证时钟/消息通道：等待一次 tick
    snap = world.wait_for_tick(seconds=timeout)
    if snap is None:
      print("[ERROR] wait_for_tick 超时，未收到 WorldSnapshot")
      return 5
    print(f"[OK] Tick: frame={snap.frame}, elapsed={snap.timestamp.elapsed_seconds:.3f}s")

    # 读取 actors 与 spectator
    actors = world.get_actors()
    print(f"[OK] Actors: {len(actors)} 个")

    spectator = world.get_spectator()
    with contextlib.suppress(Exception):
      tf = spectator.get_transform()
      print(
        f"[OK] Spectator: loc=({tf.location.x:.1f},{tf.location.y:.1f},{tf.location.z:.1f}) yaw={tf.rotation.yaw:.1f}"
      )

    # 轻量通信往返：原地切换并回滚一个不会破坏仿真的设置
    orig_no_rendering = settings.no_rendering_mode
    try:
      settings.no_rendering_mode = not orig_no_rendering
      world.apply_settings(settings)
      world.wait_for_tick(seconds=timeout)
      print(
        f"[OK] apply_settings 往返: no_rendering {orig_no_rendering} -> {settings.no_rendering_mode}"
      )
    finally:
      # 回滚
      with contextlib.suppress(Exception):
        settings.no_rendering_mode = orig_no_rendering
        world.apply_settings(settings)

    # 可选：创建一个临时 IMU 传感器验证 Spawn/Destroy
    if spawn_test:
      bp = world.get_blueprint_library().find("sensor.other.imu")
      transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
      imu = world.spawn_actor(bp, transform)
      print(f"[OK] Spawn 传感器: {imu.type_id} id={imu.id}")
      imu.destroy()
      print("[OK] Destroy 传感器: 成功")

    print("[SUCCESS] 与 CARLA 的连接与通信自检通过。")
    return 0

  except Exception as e:
    print("[ERROR] 与 CARLA 通信过程中发生异常:")
    print(e)
    return 6


if __name__ == '__main__':
  # 仅作为独立自检脚本使用，不依赖 openpilot 运行时
  import argparse
  import sys

  parser = argparse.ArgumentParser(
    description="CARLA 连接与通信快速自检（不依赖 openpilot 运行时）"
  )
  parser.add_argument("--host", default=None, help="CARLA server 主机名，默认读取 CARLA_HOST 或 127.0.0.1")
  parser.add_argument("--port", type=int, default=None, help="CARLA server 端口，默认读取 CARLA_PORT 或 2000")
  parser.add_argument("--timeout", type=float, default=None, help="超时时间（秒），默认读取 CARLA_TIMEOUT 或 5.0")
  parser.add_argument("--spawn-test", action="store_true", help="额外进行一次传感器 Spawn/Destroy 测试")

  args = parser.parse_args()
  code = test_carla_world(args.host, args.port, args.timeout, args.spawn_test)
  sys.exit(code)
