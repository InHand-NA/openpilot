import time
import numpy as np

from openpilot.common.params import Params
from openpilot.tools.sim.lib.common import SimulatorState, vec3
from openpilot.tools.sim.bridge.common import World
from openpilot.tools.sim.lib.camerad import W, H


class CarlaWorld(World):
  def __init__(self, client, high_quality, dual_camera, num_selected_spawn_point, town, carla_autopilot=False, carla_autopilot_speed=35.0):
    super().__init__(dual_camera)
    import carla

    #high_quality = False
    low_quality_layers = carla.MapLayer(carla.MapLayer.Ground | carla.MapLayer.Walls | carla.MapLayer.Decals)
    layers = carla.MapLayer.All if high_quality else low_quality_layers
    world = client.load_world(town, map_layers=layers)
    # Get the world
    #world = client.get_world()
    print("World:", world)

    # Get the map
    map = world.get_map()
    print("Map:", map)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01 #0.01
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

    self.tick_count = 0
    self.tick_batch_start = time.monotonic()

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
    self.carla_autopilot = carla_autopilot
    if carla_autopilot:
      speed_kmh = carla_autopilot_speed * 1.60934

      # TM setup with retry: load_world() resets server-side TM state,
      # and get_trafficmanager() may return a handle before TM is fully ready.
      max_retries = 3
      for attempt in range(max_retries):
        self.tm = client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.tm.set_desired_speed(self.vehicle, speed_kmh)

        # Tick a few times and verify TM is actually sending controls
        for _ in range(30):
          world.tick()
        ctl = self.vehicle.get_control()
        if ctl.throttle > 0.01:
          print(f"Carla autopilot enabled (attempt {attempt+1}), target speed: {carla_autopilot_speed:.0f} MPH ({speed_kmh:.1f} km/h)")
          break
        else:
          print(f"[WARN] TM autopilot not active after attempt {attempt+1} (throttle={ctl.throttle:.3f}), retrying...")
          self.vehicle.set_autopilot(False)
          self.tm.set_synchronous_mode(False)
      else:
        # All retries exhausted, proceed anyway — TM may activate later
        print(f"[WARN] TM autopilot may not be active after {max_retries} attempts, proceeding anyway")

  def close(self, reason: str):
    print("Closing CarlaWorld:", reason)
    if self.carla_autopilot:
      try:
        self.vehicle.set_autopilot(False)
        self.tm.set_synchronous_mode(False)
      except Exception:
        pass
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
    if self.carla_autopilot:
      # autopilot 模式：只读取 Carla 的实际控制状态，不下发指令
      self.vc = self.vehicle.get_control()
      return

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
      except Exception:
        # 回退到默认行为
        self.world.tick()
    else:
      self.world.tick()

    self.tick_count += 1
    if self.tick_count % 200 == 0:
      now = time.monotonic()
      elapsed = now - self.tick_batch_start
      sim_time = 200 * 0.01  # 200 ticks × fixed_delta_seconds(0.01s) = 2.0s
      ratio = sim_time / elapsed if elapsed > 0 else float('inf')
      print(f"[CARLA PERF] 200 ticks in {elapsed:.2f}s wall time | sim time: {sim_time:.1f}s | ratio: {ratio:.2f}x realtime")
      self.tick_batch_start = now

  def reset(self):
    import carla
    self.vehicle.set_transform(self.spawn_point)
    self.vehicle.set_target_velocity(carla.Vector3D())


