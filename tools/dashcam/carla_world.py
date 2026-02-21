import random
import threading
import time

import numpy as np


# Camera resolution matching openpilot standard
W, H = 1928, 1208


class DashcamCarlaWorld:
  """Standalone Carla world manager, no openpilot runtime dependency."""

  def __init__(self, host='127.0.0.1', port=2000, town='Town04_Opt',
               spawn_point=16, random_spawn=False, camera_pitch_deg=5.0, camera_yaw_deg=3.0,
               camera_height=1.13, high_quality=False, num_npc=20,
               wide_road_only=False, road_only=False):
    import carla

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    low_quality_layers = carla.MapLayer(carla.MapLayer.Ground | carla.MapLayer.Walls | carla.MapLayer.Decals)
    layers = carla.MapLayer.All if high_quality else low_quality_layers
    world = client.load_world(town, map_layers=layers)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.025
    settings.actor_active_distance = 150.0
    world.apply_settings(settings)
    self.weather_preset = "ClearSunset"
    world.set_weather(carla.WeatherParameters.ClearSunset)

    self.world = world
    self.sim_delta = settings.fixed_delta_seconds
    world_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # Spawn ego vehicle
    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    self.vehicle_type = vehicle_bp.id
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = world_map.get_spawn_points()
    if random_spawn:
      waypoints = world_map.generate_waypoints(2.0)
      random.shuffle(waypoints)
      self.spawn_point = None
      for wp in waypoints:
        sp = carla.Transform(wp.transform.location + carla.Location(z=0.5), wp.transform.rotation)
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
          self.spawn_point = sp
          self.vehicle = vehicle
          print(f"[RandomSpawn] Spawned at road={wp.road_id} lane={wp.lane_id} "
                + f"loc=({sp.location.x:.1f}, {sp.location.y:.1f}, {sp.location.z:.1f})")
          break
      assert self.spawn_point is not None, "Failed to spawn at any random waypoint"
    else:
      assert len(spawn_points) > spawn_point, \
        f'No spawn point {spawn_point}, try 0-{len(spawn_points)-1}'
      self.spawn_point = spawn_points[spawn_point]
      self.vehicle = world.spawn_actor(vehicle_bp, self.spawn_point)

    physics_control = self.vehicle.get_physics_control()
    physics_control.mass = 2326
    physics_control.torque_curve = [carla.Vector2D(20.0, 500.0), carla.Vector2D(5000.0, 500.0)]
    physics_control.gear_switch_time = 0.0
    self.vehicle.apply_physics_control(physics_control)

    self.carla_objects = []

    # Camera transform (extrinsics relative to vehicle origin)
    self.camera_offset_x = 0.8
    self.camera_height = camera_height
    self.camera_pitch_deg = camera_pitch_deg
    self.camera_yaw_deg = camera_yaw_deg
    transform = carla.Transform(
      carla.Location(x=0.8, z=camera_height),
      carla.Rotation(pitch=camera_pitch_deg, yaw=camera_yaw_deg))

    # Image buffers and synchronization
    self.image_lock = threading.Lock()
    self.road_image = None
    self.wide_road_image = None
    self._new_frame = False

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      blueprint.set_attribute('sensor_tick', str(1.0 / sensor_fps))
      if not high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=self.vehicle)
      camera.listen(callback)
      return camera

    sensor_fps = 20.0
    def cam_info(fov):
      focal = (W / 2.0) / np.tan(np.radians(fov / 2.0))
      return {"fov_deg": fov, "width": W, "height": H, "focal_length": round(focal, 1), "fps": sensor_fps}

    self.wide_road_only = wide_road_only
    self.road_only = road_only
    self.cameras_info = {}
    if wide_road_only:
      self.road_camera = None
      self.wide_road_camera = create_camera(fov=120, callback=self._cam_callback_wide)
      self.carla_objects = [self.wide_road_camera, self.vehicle]
      self.cameras_info["wide_road"] = cam_info(120)
    elif road_only:
      self.road_camera = create_camera(fov=40, callback=self._cam_callback_road)
      self.wide_road_camera = None
      self.carla_objects = [self.road_camera, self.vehicle]
      self.cameras_info["road"] = cam_info(40)
    else:
      self.road_camera = create_camera(fov=40, callback=self._cam_callback_road)
      self.wide_road_camera = create_camera(fov=120, callback=self._cam_callback_wide)
      self.carla_objects = [self.road_camera, self.wide_road_camera, self.vehicle]
      self.cameras_info["road"] = cam_info(40)
      self.cameras_info["wide_road"] = cam_info(120)

    # Traffic manager
    self.tm = client.get_trafficmanager()
    self.tm.set_synchronous_mode(True)
    self.tm.set_respawn_dormant_vehicles(True)
    self.tm.set_boundaries_respawn_dormant_vehicles(25.0, 100.0)

    # Enable autopilot with retry
    speed_kmh = 35.0 * 1.60934
    max_retries = 3
    for attempt in range(max_retries):
      self.vehicle.set_autopilot(True, self.tm.get_port())
      self.tm.set_desired_speed(self.vehicle, speed_kmh)
      for _ in range(30):
        world.tick()
      ctl = self.vehicle.get_control()
      if ctl.throttle > 0.01:
        print(f"Carla autopilot enabled (attempt {attempt+1})")
        break
      else:
        print(f"[WARN] TM autopilot not active after attempt {attempt+1}, retrying...")
        self.vehicle.set_autopilot(False)
    else:
      print(f"[WARN] TM autopilot may not be active after {max_retries} attempts, proceeding anyway")

    # Spawn NPC vehicles
    self.npc_vehicles = []
    npc_bps = [bp for bp in blueprint_library.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    available_points = [sp for sp in spawn_points
                        if sp.location.distance(self.spawn_point.location) > 2.0]
    random.shuffle(available_points)
    for sp in available_points[:num_npc]:
      bp = random.choice(npc_bps)
      if bp.has_attribute('color'):
        bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
      npc = world.try_spawn_actor(bp, sp)
      if npc is not None:
        npc.set_autopilot(True, self.tm.get_port())
        self.npc_vehicles.append(npc)
    self.carla_objects.extend(self.npc_vehicles)
    print(f"Spawned {len(self.npc_vehicles)} NPC vehicles")

    self.tick_count = 0
    self.tick_batch_start = time.monotonic()

  def _carla_image_to_rgb(self, image):
    rgb = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    rgb = np.reshape(rgb, (H, W, 4))
    return np.ascontiguousarray(rgb[:, :, [0, 1, 2]])

  def _cam_callback_road(self, image):
    with self.image_lock:
      self.road_image = self._carla_image_to_rgb(image)
      self._new_frame = True

  def _cam_callback_wide(self, image):
    with self.image_lock:
      self.wide_road_image = self._carla_image_to_rgb(image)
      if self.wide_road_only:
        self._new_frame = True

  def get_frame(self):
    """Get latest camera RGB frames. Returns (road, wide) or (None, None) if no new frame."""
    with self.image_lock:
      if not self._new_frame:
        return None, None
      self._new_frame = False
      road = self.road_image.copy() if self.road_image is not None else None
      wide = self.wide_road_image.copy() if self.wide_road_image is not None else None
      return road, wide

  def get_map(self):
    """Return the Carla map object."""
    return self.world.get_map()

  def get_world(self):
    """Return the Carla world object (for actor queries)."""
    return self.world

  def get_vehicle(self):
    """Return the ego vehicle actor."""
    return self.vehicle

  def get_clip_metadata(self, town):
    """Return clip-level metadata dict for JSON serialization."""
    weather = self.world.get_weather()
    sp = self.spawn_point
    return {
      "town": town,
      "weather": {
        "preset": self.weather_preset,
        "sun_altitude_angle": weather.sun_altitude_angle,
        "sun_azimuth_angle": weather.sun_azimuth_angle,
        "cloudiness": weather.cloudiness,
        "precipitation": weather.precipitation,
        "fog_density": weather.fog_density,
      },
      "vehicle": {
        "type": self.vehicle_type,
        "spawn_location": {
          "x": sp.location.x,
          "y": sp.location.y,
          "z": sp.location.z,
        },
        "spawn_rotation": {
          "pitch": sp.rotation.pitch,
          "yaw": sp.rotation.yaw,
          "roll": sp.rotation.roll,
        },
      },
      "cameras": {
        name: {
          "fov_deg": info["fov_deg"],
          "image_width": info["width"],
          "image_height": info["height"],
          "focal_length": info["focal_length"],
          "fps": info["fps"],
          "extrinsics": {
            "x": self.camera_offset_x,
            "y": 0.0,
            "z": self.camera_height,
            "pitch_deg": self.camera_pitch_deg,
            "yaw_deg": self.camera_yaw_deg,
            "roll_deg": 0.0,
          },
        }
        for name, info in self.cameras_info.items()
      },
      "simulation": {
        "fixed_delta_seconds": self.sim_delta,
        "fps": 1.0 / self.sim_delta,
        "num_npc_vehicles": len(self.npc_vehicles),
      },
    }

  def get_vehicle_transform(self):
    """Return the vehicle's carla.Transform."""
    return self.vehicle.get_transform()

  def get_vehicle_speed(self):
    """Get 3D speed scalar in m/s."""
    v = self.vehicle.get_velocity()
    return np.sqrt(v.x**2 + v.y**2 + v.z**2)

  def tick(self):
    """Advance simulation by one step."""
    self.world.tick()
    self.tick_count += 1
    delta = self.world.get_settings().fixed_delta_seconds
    batch = max(1, int(2.0 / delta))  # ~2s of sim time per log
    if self.tick_count % batch == 0:
      now = time.monotonic()
      elapsed = now - self.tick_batch_start
      sim_time = batch * delta
      ratio = sim_time / elapsed if elapsed > 0 else float('inf')
      print(f"[CARLA PERF] {batch} ticks in {elapsed:.2f}s | sim: {sim_time:.1f}s | ratio: {ratio:.2f}x realtime")
      self.tick_batch_start = now

  def close(self):
    """Destroy all actors and clean up (idempotent)."""
    if not self.carla_objects:
      return
    try:
      self.vehicle.set_autopilot(False)
    except Exception:
      pass
    # Stop camera listeners before destroying to avoid C++ callback crashes
    for cam in [c for c in [self.road_camera, self.wide_road_camera] if c is not None]:
      try:
        if cam is not None and cam.is_listening:
          cam.stop()
      except Exception:
        pass
    try:
      self.tm.set_synchronous_mode(False)
    except Exception:
      pass
    # Tick once to let Carla process the stop commands
    try:
      settings = self.world.get_settings()
      settings.synchronous_mode = False
      self.world.apply_settings(settings)
    except Exception:
      pass
    for s in self.carla_objects:
      if s is not None:
        try:
          s.destroy()
        except Exception:
          pass
    self.carla_objects = []
