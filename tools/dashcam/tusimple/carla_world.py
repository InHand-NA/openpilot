"""TuSimple Carla world — H0 reference stereo + H1~H6 mono cameras.

H0 (openpilot narrow+wide at 1.22m) provides modeld 3D annotation baseline.
H1~H6 (1920×1080, FOV=120°) at various heights provide TuSimple training images.

All 8 cameras share the same pitch/yaw rotation; only height differs (H1~H6).

Usage:
  from openpilot.tools.dashcam.tusimple.config import CameraSlotConfig
  world = TuSimpleCarlaWorld(
      mono_heights=[CameraSlotConfig('H1', 1.22), CameraSlotConfig('H6', 3.0)],
      town='Town04',
  )
  world.tick()
  frames = world.get_frames()  # blocks until all cameras report the same frame
  world.close()
"""

import random
import threading
import time

import numpy as np

from openpilot.tools.dashcam.tusimple.config import (
  H0_H,
  H0_HEIGHT,
  H0_NARROW_FOV,
  H0_W,
  H0_WIDE_FOV,
  HEIGHT_DEFS,
  MONO_H,
  MONO_HFOV,
  MONO_W,
  CameraSlotConfig,
  compute_crop_params,
)


class TuSimpleCarlaWorld:
  """Carla world with H0 reference cameras + H1~H6 mono cameras.

  H0 (openpilot narrow+wide at 1.22m) provides modeld 3D annotation.
  H1~H6 (1920×1080, FOV=120°) at various heights provide TuSimple training images.

  Args:
    host: Carla server hostname
    port: Carla server port
    town: map name (e.g. 'Town04')
    weather: weather preset name
    spawn_point: index into world's spawn_points list
    random_spawn: if True, pick a random drivable waypoint
    camera_pitch_deg: camera pitch angle (deg, positive=nose down)
    camera_yaw_deg: camera yaw angle (deg)
    camera_forward_offset_m: forward offset from vehicle center
    mono_heights: list of CameraSlotConfig for H1~H6 mono cameras
    num_npc: number of NPC vehicles
    high_quality: enable Carla post-processing effects
    speed_range: (min, max) target speed range in km/h
    speed_interval: (min, max) interval in seconds between speed changes
  """

  def __init__(
    self,
    host: str = '127.0.0.1',
    port: int = 2000,
    town: str = 'Town04',
    weather: str = 'ClearNoon',
    spawn_point: int = 16,
    random_spawn: bool = False,
    camera_pitch_deg: float = 5.0,
    camera_yaw_deg: float = 0.0,
    camera_forward_offset_m: float = 0.8,
    mono_heights: list[CameraSlotConfig] | None = None,
    num_npc: int = 40,
    high_quality: bool = False,
    speed_range: tuple[float, float] = (40.0, 100.0),
    speed_interval: tuple[float, float] = (8.0, 20.0),
  ):
    import carla

    if mono_heights is None:
      mono_heights = [
        CameraSlotConfig('H1', HEIGHT_DEFS['H1']),
        CameraSlotConfig('H6', HEIGHT_DEFS['H6']),
      ]

    self._mono_slots = mono_heights
    self.camera_pitch_deg = camera_pitch_deg
    self.camera_yaw_deg = camera_yaw_deg
    self.camera_forward_offset_m = camera_forward_offset_m

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    self._client = client

    low_quality_layers = carla.MapLayer(carla.MapLayer.Ground | carla.MapLayer.Walls | carla.MapLayer.Decals)
    layers = carla.MapLayer.All if high_quality else low_quality_layers
    world = client.load_world(town, map_layers=layers)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05   # 20 FPS
    settings.actor_active_distance = 200.0
    world.apply_settings(settings)
    self.sim_delta = settings.fixed_delta_seconds

    # Apply weather
    self.weather_name = weather
    self._apply_weather(world, weather)

    self.world = world
    world_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # Spawn ego vehicle (Tesla)
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
          print(f"[RandomSpawn] spawned at road={wp.road_id} lane={wp.lane_id} "
                f"loc=({sp.location.x:.1f}, {sp.location.y:.1f}, {sp.location.z:.1f})")
          break
      assert self.spawn_point is not None, "Failed to spawn at any random waypoint"
    else:
      assert len(spawn_points) > spawn_point, \
        f"No spawn point {spawn_point}, try 0-{len(spawn_points) - 1}"
      self.spawn_point = spawn_points[spawn_point]
      self.vehicle = world.spawn_actor(vehicle_bp, self.spawn_point)

    physics_control = self.vehicle.get_physics_control()
    physics_control.mass = 2326
    physics_control.torque_curve = [carla.Vector2D(20.0, 500.0), carla.Vector2D(5000.0, 500.0)]
    physics_control.gear_switch_time = 0.0
    self.vehicle.apply_physics_control(physics_control)

    # Camera image buffer: heterogeneous structure
    #   H0: {'road': (frame_id, rgb), 'wide': (frame_id, rgb)}
    #   H1~H6: {'mono': (frame_id, rgb)}
    self._latest: dict[str, dict[str, tuple[int, np.ndarray] | None]] = {
      'H0': {'road': None, 'wide': None},
      **{slot.tag: {'mono': None} for slot in self._mono_slots},
    }
    self._lock = threading.Lock()

    sensor_fps = 20.0
    self._sensors: list = []

    def create_camera(fov, width, height, transform, callback):
      bp = blueprint_library.find('sensor.camera.rgb')
      bp.set_attribute('image_size_x', str(width))
      bp.set_attribute('image_size_y', str(height))
      bp.set_attribute('fov', str(fov))
      bp.set_attribute('sensor_tick', str(1.0 / sensor_fps))
      if not high_quality:
        bp.set_attribute('enable_postprocess_effects', 'False')
      cam = world.spawn_actor(bp, transform, attach_to=self.vehicle)
      cam.listen(callback)
      return cam

    # H0 reference cameras at fixed height 1.22m
    h0_transform = carla.Transform(
      carla.Location(x=camera_forward_offset_m, z=H0_HEIGHT),
      carla.Rotation(pitch=-camera_pitch_deg, yaw=-camera_yaw_deg),
    )
    h0_road = create_camera(
      fov=H0_NARROW_FOV, width=H0_W, height=H0_H,
      transform=h0_transform,
      callback=self._make_callback('H0', 'road', H0_H, H0_W),
    )
    h0_wide = create_camera(
      fov=H0_WIDE_FOV, width=H0_W, height=H0_H,
      transform=h0_transform,
      callback=self._make_callback('H0', 'wide', H0_H, H0_W),
    )
    self._sensors.extend([h0_road, h0_wide])

    # H1~H6 mono cameras at varying heights
    for slot in self._mono_slots:
      t = carla.Transform(
        carla.Location(x=camera_forward_offset_m, z=slot.height),
        carla.Rotation(pitch=-camera_pitch_deg, yaw=-camera_yaw_deg),
      )
      mono_cam = create_camera(
        fov=MONO_HFOV, width=MONO_W, height=MONO_H,
        transform=t,
        callback=self._make_callback(slot.tag, 'mono', MONO_H, MONO_W),
      )
      self._sensors.append(mono_cam)

    # Traffic manager
    self.tm = client.get_trafficmanager()
    self.tm.set_synchronous_mode(True)
    self.tm.set_respawn_dormant_vehicles(True)
    self.tm.set_boundaries_respawn_dormant_vehicles(25.0, 100.0)

    self.speed_range = speed_range
    self.speed_interval = speed_interval
    speed_kmh = random.uniform(*speed_range)

    max_retries = 3
    for attempt in range(max_retries):
      self.vehicle.set_autopilot(True, self.tm.get_port())
      self.tm.set_desired_speed(self.vehicle, speed_kmh)
      for _ in range(30):
        world.tick()
      ctl = self.vehicle.get_control()
      if ctl.throttle > 0.01:
        print(f"Carla autopilot enabled (attempt {attempt + 1})")
        break
      else:
        print(f"[WARN] autopilot not active after attempt {attempt + 1}, retrying...")
        self.vehicle.set_autopilot(False)
    else:
      print(f"[WARN] autopilot may not be active after {max_retries} attempts, proceeding anyway")

    # Spawn NPC vehicles
    self.npc_vehicles = []
    npc_bps = [bp for bp in blueprint_library.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) >= 4]
    available_pts = [sp for sp in spawn_points
                     if sp.location.distance(self.spawn_point.location) > 2.0]
    random.shuffle(available_pts)
    for sp in available_pts[:num_npc]:
      bp = random.choice(npc_bps)
      if bp.has_attribute('color'):
        bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
      npc = world.try_spawn_actor(bp, sp)
      if npc is not None:
        npc.set_autopilot(True, self.tm.get_port())
        self.npc_vehicles.append(npc)
    print(f"Spawned {len(self.npc_vehicles)} NPC vehicles")

    self._all_actors = self._sensors + self.npc_vehicles + [self.vehicle]

    self.tick_count = 0
    self._tick_batch_start = time.monotonic()
    interval = random.uniform(*speed_interval)
    self._next_speed_change_tick = int(interval / self.sim_delta)

  @staticmethod
  def _apply_weather(world, weather_name: str):
    import carla
    presets = {
      'ClearNoon':     carla.WeatherParameters.ClearNoon,
      'ClearSunset':   carla.WeatherParameters.ClearSunset,
      'CloudyNoon':    carla.WeatherParameters.CloudyNoon,
      'WetNoon':       carla.WeatherParameters.WetNoon,
      'WetSunset':     carla.WeatherParameters.WetSunset,
      'MidRainSunset': carla.WeatherParameters.MidRainSunset,
      'SoftRainNoon':  carla.WeatherParameters.SoftRainNoon,
    }
    world.set_weather(presets.get(weather_name, carla.WeatherParameters.ClearNoon))

  def _make_callback(self, tag: str, key: str, img_h: int, img_w: int):
    """Return a Carla sensor callback that stores (frame_id, rgb) under _latest[tag][key]."""
    def callback(image):
      rgb = np.frombuffer(image.raw_data, dtype=np.uint8)
      rgb = np.ascontiguousarray(rgb.reshape((img_h, img_w, 4))[:, :, :3])  # BGRA→BGR (Carla sends BGR-ordered)
      with self._lock:
        self._latest[tag][key] = (image.frame, rgb)
    return callback

  def get_frames(self) -> dict | None:
    """Return synchronized frames when ALL cameras have same frame_id.

    Returns:
      {
        'h0_road': np.ndarray,  # (1208, 1928, 3) RGB
        'h0_wide': np.ndarray,  # (1208, 1928, 3) RGB
        'mono': {
          'H1': np.ndarray,     # (1080, 1920, 3) RGB
          ...
        }
      }
      or None if not yet synchronized.
    """
    with self._lock:
      # Collect all entries: H0 road/wide + mono cameras
      all_entries = []
      h0 = self._latest['H0']
      all_entries.append(h0['road'])
      all_entries.append(h0['wide'])
      for slot in self._mono_slots:
        all_entries.append(self._latest[slot.tag]['mono'])

      if any(e is None for e in all_entries):
        return None  # some cameras haven't sent data yet
      frame_ids = [e[0] for e in all_entries]
      if len(set(frame_ids)) != 1:
        return None  # frame IDs don't match — wait for lagging callbacks

      return {
        'h0_road': h0['road'][1].copy(),
        'h0_wide': h0['wide'][1].copy(),
        'mono': {
          slot.tag: self._latest[slot.tag]['mono'][1].copy()
          for slot in self._mono_slots
        },
      }

  def tick(self) -> None:
    """Advance simulation by one step and update speed if needed."""
    self.world.tick()
    self.tick_count += 1
    self._update_speed()
    batch = max(1, int(2.0 / self.sim_delta))
    if self.tick_count % batch == 0:
      now = time.monotonic()
      elapsed = now - self._tick_batch_start
      sim_time = batch * self.sim_delta
      ratio = sim_time / elapsed if elapsed > 0 else float('inf')
      print(f"[CARLA PERF] {batch} ticks in {elapsed:.2f}s | sim={sim_time:.1f}s | ratio={ratio:.2f}x")
      self._tick_batch_start = now

  def _update_speed(self):
    if self.tick_count < self._next_speed_change_tick:
      return
    new_speed = random.uniform(*self.speed_range)
    self.tm.set_desired_speed(self.vehicle, new_speed)
    interval = random.uniform(*self.speed_interval)
    self._next_speed_change_tick = self.tick_count + int(interval / self.sim_delta)

  def get_vehicle_speed(self) -> float:
    """Return ego vehicle speed in m/s."""
    v = self.vehicle.get_velocity()
    return float(np.sqrt(v.x**2 + v.y**2 + v.z**2))

  def get_vehicle_transform(self):
    """Return ego vehicle carla.Transform."""
    return self.vehicle.get_transform()

  def get_clip_metadata(self, session_id: str) -> dict:
    """Build clip_info.json dict for this session."""
    sp = self.spawn_point
    crop = compute_crop_params()
    crop_rect = crop['crop_rect']
    return {
      'session_id': session_id,
      'map': self.world.get_map().name.split('/')[-1],
      'weather': self.weather_name,
      'camera': {
        'pitch_deg': self.camera_pitch_deg,
        'yaw_deg': self.camera_yaw_deg,
        'forward_offset_m': self.camera_forward_offset_m,
      },
      'h0_camera': {
        'tag': 'H0',
        'height': H0_HEIGHT,
        'narrow': {'width': H0_W, 'height': H0_H, 'fov': H0_NARROW_FOV, 'focal': 2648.0},
        'wide':   {'width': H0_W, 'height': H0_H, 'fov': H0_WIDE_FOV, 'focal': 567.0},
      },
      'mono_camera': {
        'width': MONO_W, 'height': MONO_H, 'fov': MONO_HFOV,
        'focal': round(float(MONO_W / 2 / np.tan(np.radians(MONO_HFOV / 2))), 1),
        'model': 'pinhole',
      },
      'tusimple_output': {
        'width': 1280, 'height': 720,
        'crop_hfov': crop['crop_hfov'],
        'nominal_pitch': 4.0,
        'crop_rect': list(crop_rect),
        'effective_focal': round(float(crop['effective_focal']), 1),
      },
      'heights': {slot.tag: slot.height for slot in self._mono_slots},
      'simulation': {
        'fps': 1.0 / self.sim_delta,
        'fixed_delta_seconds': self.sim_delta,
        'num_npc': len(self.npc_vehicles),
      },
      'spawn': {
        'x': sp.location.x,
        'y': sp.location.y,
        'z': sp.location.z,
        'yaw': sp.rotation.yaw,
      },
    }

  def close(self) -> None:
    """Destroy all actors and restore async mode (idempotent).

    Correct shutdown order:
      1. Disable vehicle autopilot
      2. Stop all camera listeners
      3. Disable TM sync mode
      4. Tick once to flush any pending sensor callbacks
      5. Batch-destroy all actors via apply_batch_sync
      6. Switch world to async mode LAST
    """
    if not self._all_actors:
      return

    import carla

    # 1. Disable autopilot
    try:
      self.vehicle.set_autopilot(False)
    except Exception:
      pass

    # 2. Stop camera listeners
    for cam in self._sensors:
      try:
        cam.stop()
      except Exception:
        pass

    # 3. Disable traffic manager sync
    try:
      self.tm.set_synchronous_mode(False)
    except Exception:
      pass

    # 4. One final tick in sync mode to flush pending callbacks
    try:
      self.world.tick()
    except Exception:
      pass
    try:
      settings = self.world.get_settings()
      settings.synchronous_mode = False
      self.world.apply_settings(settings)
    except Exception:
      pass

    # 5. Batch-destroy all actors at once
    try:
      batch = [carla.command.DestroyActor(a) for a in self._all_actors]
      self._client.apply_batch_sync(batch, False)
    except Exception:
      for actor in self._all_actors:
        try:
          actor.destroy()
        except Exception:
          pass

    self._all_actors = []
    print("[TuSimpleCarlaWorld] closed.")
