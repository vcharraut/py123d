# Checks a town's carla_maps/speed_signs.json rows against CARLA: moves a car along random routes in 1 m steps and
# reports each speed-limit change that no row of that lane, limit and s (within 3 m) explains. A change up to 1.5 m
# before a road end is explained by a row at the start of the next road (the car takes the limit before its centre
# crosses the border). Run on a server restarted for the town (see export_speed_signs.py). Python 3.12 + carla wheel.
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/check_speed_signs.py <speed_signs.json> <port> Town11 60 2000
import json, random, sys, time
import carla

signs, port, town, n_routes, route_m = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rows = {}
for road, lane, s, limit in json.load(open(signs))[town]:
    rows.setdefault((road, lane), []).append((s, limit))
client = carla.Client("localhost", port); client.set_timeout(1200)
world = client.get_world()
if not world.get_map().name.endswith(town):
    world = client.load_world(town)
settings = world.get_settings()
settings.synchronous_mode, settings.fixed_delta_seconds, settings.actor_active_distance = True, 0.05, 1e6
world.apply_settings(settings)
cmap = world.get_map()
bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
air = lambda tf: carla.Transform(tf.location + carla.Location(z=150.0 + time.time() % 100), tf.rotation)
bp.set_attribute("role_name", "hero")
hero = world.spawn_actor(bp, air(cmap.get_spawn_points()[0])); hero.set_simulate_physics(False)
bp.set_attribute("role_name", "probe")
random.seed(0)
changes, misses = 0, []
for k, start in enumerate(random.sample(cmap.get_spawn_points(), n_routes)):
    print(f"route {k}/{n_routes}: {changes} changes, {len(misses)} unexplained", flush=True)
    car = world.spawn_actor(bp, air(start)); car.set_simulate_physics(False)
    world.tick()
    wp, limit = cmap.get_waypoint(start.location), 30.0
    for _ in range(route_m):
        car.set_transform(carla.Transform(wp.transform.location + carla.Location(z=0.3), wp.transform.rotation))
        world.tick(); world.tick()
        new = car.get_speed_limit()
        if new != limit:
            changes += 1
            if not any(lim == new and abs(s - wp.s) <= 3.0 for s, lim in rows.get((wp.road_id, wp.lane_id), [])):
                misses.append([wp.road_id, wp.lane_id, round(wp.s, 1), limit, new])
            limit = new
        nxt = wp.next(1.0)
        if not nxt:
            break
        wp = random.choice(nxt)
    car.destroy(); world.tick()
print(town, changes, "limit changes, unexplained [road, lane, s, from, to]:", misses, flush=True)
hero.destroy(); world.tick()
