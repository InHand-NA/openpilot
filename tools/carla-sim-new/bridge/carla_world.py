import time
import numpy as np

from openpilot.common.params import Params
from openpilot.tools.sim.lib.common import SimulatorState, vec3
from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.lib.common import World, W, H


class CarlaWorld(World):
  def __init__(self, status_q, client, high_quality, dual_camera, num_selected_spawn_point, town):
    super().__init__(dual_camera)
    import carla

    self.status_q = status_q
    self.params = Params()

    low_quality_layers = carla.MapLayer(carla.MapLayer.Ground | carla.MapLayer.Walls | carla.MapLayer.Decals)
    layers = carla.MapLayer.All if high_quality else low_quality_layers

    world = client.load_world(town, map_layers=layers)

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.01
    settings.synchronous_mode = True
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    self.world = world
    self.world_map = world.get_map()

    blueprint_library = world.get_blueprint_library()

    # prefer tesla or use first vehicle blueprint as fallback
    vehicle_bps = blueprint_library.filter('vehicle.tesla.*')
    vehicle_bp = vehicle_bps[1] if len(vehicle_bps) > 1 else blueprint_library.filter('vehicle.*')[0]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = self.world_map.get_spawn_points()
    assert len(spawn_points) > num_selected_spawn_point, \
      f"No spawn point {num_selected_spawn_point}, valid range: 0..{len(spawn_points)-1} in {town}"
    self.spawn_point = spawn_points[num_selected_spawn_point]
    self.vehicle = world.spawn_actor(vehicle_bp, self.spawn_point)

    physics_control = self.vehicle.get_physics_control()
    physics_control.mass = 2326
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    self.vehicle.apply_physics_control(physics_control)

    self.max_steer_angle: float = self.vehicle.get_physics_control().wheels[0].max_steer_angle
    self.steer_ratio = 15

    self.vc: carla.VehicleControl = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

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
    self.road_wide_camera = create_camera(fov=120, callback=self.cam_callback_wide_road) if dual_camera else None

    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    self.imu = world.spawn_actor(imu_bp, transform, attach_to=self.vehicle)

    gps_bp = blueprint_library.find('sensor.other.gnss')
    self.gps = world.spawn_actor(gps_bp, transform, attach_to=self.vehicle)
    self.params.put_bool("UbloxAvailable", True)

    self.carla_objects = [self.imu, self.gps, self.road_camera, self.road_wide_camera, self.vehicle]

    # book-keeping
    self.last_tick = time.monotonic()

    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "started"))

  def close(self, reason: str):
    try:
      for s in self.carla_objects:
        if s is not None:
          try:
            s.destroy()
          except Exception:
            pass
    finally:
      self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))

  def carla_image_to_rgb(self, image):
    rgb = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    rgb = np.reshape(rgb, (H, W, 4))
    return np.ascontiguousarray(rgb[:, :, [0, 1, 2]])

  def cam_callback_road(self, image):
    self.road_image = self.carla_image_to_rgb(image)
    self.image_lock.release()

  def cam_callback_wide_road(self, image):
    self.wide_road_image = self.carla_image_to_rgb(image)
    self.image_lock.release()

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    self.vc.throttle = throttle_out
    steer_carla = steer_angle * -1 / (self.max_steer_angle * self.steer_ratio)
    steer_carla = float(np.clip(steer_carla, -1, 1))
    self.vc.steer = steer_carla
    self.vc.brake = brake_out
    self.vehicle.apply_control(self.vc)

  def read_state(self):
    pass

  def read_sensors(self, simulator_state: SimulatorState):
    # IMU bearing from vehicle yaw
    tf = self.vehicle.get_transform()
    simulator_state.imu.bearing = float(tf.rotation.yaw % 360)

    # IMU linear acc and gyro from vehicle kinematics
    acc = self.vehicle.get_acceleration()
    gyro = self.vehicle.get_angular_velocity()
    simulator_state.imu.accelerometer = vec3(acc.x, acc.y, acc.z)
    simulator_state.imu.gyroscope = vec3(gyro.x, gyro.y, gyro.z)

    # Approximate GPS from world XY for stability
    loc = self.vehicle.get_location()
    simulator_state.gps.from_xy([loc.x, loc.y])

    # Velocity and steering angle
    v = self.vehicle.get_velocity()
    simulator_state.velocity = vec3(v.x, v.y, v.z)
    simulator_state.steering_angle = float(self.vehicle.get_control().steer * self.max_steer_angle)
    simulator_state.valid = True

  def read_cameras(self):
    pass

  def tick(self):
    # In synchronous mode, tick advances the world
    self.world.tick()

  def reset(self):
    import carla
    self.vehicle.set_transform(self.spawn_point)
    self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
    self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
