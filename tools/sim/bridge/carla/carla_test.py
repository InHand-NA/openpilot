import carla
import random

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
    # 将观众移动到车辆上方以便查看
    t = actor.get_transform()
    loc = t.location
    spectator.set_transform(carla.Transform(
      carla.Location(x=loc.x, y=loc.y, z=loc.z + 25.0),
      carla.Rotation(pitch=-90.0)
    ))
  else:
    print("Failed to spawn a vehicle at a random road point")



if __name__ == "__main__":
  main()
