import carla
import random
import contextlib

def main():
  client = carla.Client('localhost', 2000)
  client.set_timeout(10.0)

  # Get the world
  world = client.get_world()
  print("World:", world)

  # Get the map
  map = world.get_map()
  print("Map:", map)

  # Get the weather
  weather = world.get_weather()
  print("Weather:", weather)

  # Get the spectator
  spectator = world.get_spectator()
  print("Spectator:", spectator)

  # Get the actors
  actors = world.get_actors()
  print("Actors:", actors)

  # 在随机道路点生成一辆车辆
  bp_lib = world.get_blueprint_library()
  vehicle_bps = bp_lib.filter('vehicle.*')
  spawn_points = map.get_spawn_points()

  actor = None
  if spawn_points and vehicle_bps:
    random.shuffle(spawn_points)
    # 限制尝试次数以提高速度
    for sp in spawn_points[:16]:
      bp = random.choice(vehicle_bps)
      if bp.has_attribute('role_name'):
        bp.set_attribute('role_name', 'hero')
      actor = world.try_spawn_actor(bp, sp)
      if actor is not None:
        break

  if actor is not None:
    print("Spawned vehicle:", actor)
    # 切换到同步模式，并配置固定步长
    original_settings = world.get_settings()
    new_settings = carla.WorldSettings(
      no_rendering_mode=original_settings.no_rendering_mode,
      synchronous_mode=True,
      fixed_delta_seconds=0.05,
      substepping=original_settings.substepping,
      max_substep_delta_time=original_settings.max_substep_delta_time,
      max_substeps=original_settings.max_substeps,
    )

    tm = None
    with contextlib.suppress(Exception):
      tm = client.get_trafficmanager()
      tm.set_synchronous_mode(True)

    world.apply_settings(new_settings)

    # 开启自动驾驶（绑定到 Traffic Manager 端口以确保同步控制）
    with contextlib.suppress(Exception):
      if tm is not None:
        actor.set_autopilot(True, tm.get_port())
      else:
        actor.set_autopilot(True)

    # 使用 tick 驱动同步仿真，每步更新观众视角
    try:
      while True:
        world.tick()
        t = actor.get_transform()
        loc = t.location
        spectator.set_transform(carla.Transform(
          carla.Location(x=loc.x, y=loc.y, z=loc.z + 25.0),
          carla.Rotation(pitch=-90.0)
        ))
    except KeyboardInterrupt:
      pass
    finally:
      with contextlib.suppress(Exception):
        actor.destroy()
      with contextlib.suppress(Exception):
        if tm is not None:
          tm.set_synchronous_mode(False)
      with contextlib.suppress(Exception):
        world.apply_settings(original_settings)
  else:
    print("Failed to spawn a vehicle at a random road point")



if __name__ == "__main__":
  main()
