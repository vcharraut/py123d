# Exports the speed signs of a CARLA town to carla_maps/speed_signs.json: [road id, lane id, s, limit in km/h] per lane
# a traffic.speed_limit actor acts on, at the first 2 m grid point in driving direction where a car put there takes the
# sign's limit. A spawned car has CARLA's default 30 km/h, so for a 30 km/h sign it first takes another limit on the
# nearest other sign. Large maps (Town11, 12, 13) put actors farther than actor_active_distance from a hero vehicle to
# sleep, and sleeping triggers never fire: one hero stays spawned and the distance is raised, so triggers fire without
# loaded tiles. Restart the server for each town (a second load_world crashes CARLA 0.9.16). Python 3.12 + carla wheel.
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/export_speed_signs.py <out.json> <port> Town11
import json, os, sys, time
import carla
import numpy as np

TICKS, RADIUS, STEP = 3, 6.0, 2.0  # [-] after a teleport, [m] beyond the trigger's largest half extent, [m] grid
out, port, town = sys.argv[1], int(sys.argv[2]), sys.argv[3]
client = carla.Client("localhost", port); client.set_timeout(1200)
world = client.get_world()
if not world.get_map().name.endswith(town):
    world = client.load_world(town)
settings = world.get_settings()
settings.synchronous_mode, settings.fixed_delta_seconds, settings.actor_active_distance = True, 0.05, 1e6
world.apply_settings(settings)
cmap = world.get_map()
bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
# Spawned in the air, so spawns never collide with the map or a car left by an earlier run.
air = lambda tf: carla.Transform(tf.location + carla.Location(z=150.0 + time.time() % 100), tf.rotation)
bp.set_attribute("role_name", "hero")
hero = world.spawn_actor(bp, air(cmap.get_spawn_points()[0])); hero.set_simulate_physics(False)
for _ in range(20):
    world.tick()

signs = [(float(a.type_id.split(".")[-1]), a.get_transform().transform(a.trigger_volume.location), a.trigger_volume.extent)
         for a in world.get_actors().filter("traffic.speed_limit.*")]
wps = cmap.generate_waypoints(STEP)
xyz = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in wps])
primers = [w.transform for limit, loc, _ in signs if limit != 30.0 and (w := cmap.get_waypoint(loc)) is not None]
primer_xy = np.array([[p.location.x, p.location.y] for p in primers]).reshape(-1, 2)
print(town, "signs", len(signs), "waypoints", len(wps), "primers", len(primers), flush=True)
bp.set_attribute("role_name", "probe")
car = None

def place(tf):
    car.set_transform(carla.Transform(tf.location + carla.Location(z=0.3), tf.rotation))
    for _ in range(TICKS):
        world.tick()
    return car.get_speed_limit()

def reset(tf, limit):
    global car
    if car is not None:
        car.destroy()
    car = world.spawn_actor(bp, air(tf)); car.set_simulate_physics(False)
    world.tick()
    if limit != 30.0:
        return True
    nearest = np.argsort(np.linalg.norm(primer_xy - [tf.location.x, tf.location.y], axis=1))[:5]
    return any(place(primers[i]) != 30.0 for i in nearest)

rows, unprimed = [], 0
for n, (limit, loc, ext) in enumerate(signs):
    near = np.flatnonzero((np.hypot(xyz[:, 0] - loc.x, xyz[:, 1] - loc.y) < max(ext.x, ext.y) + RADIUS)
                          & (np.abs(xyz[:, 2] - loc.z) < 3.0))
    lane_s = {}
    for i in near:
        lane_s.setdefault((wps[i].road_id, wps[i].lane_id), set()).add(STEP * round(wps[i].s / STEP))
    for (road, lane), s_set in sorted(lane_s.items()):
        grid = np.arange(min(s_set) - STEP, max(s_set) + 2 * STEP, STEP)
        for s in grid[grid >= 0.0][:: 1 if lane < 0 else -1]:  # driving direction: lanes with id < 0 follow s
            wp = cmap.get_waypoint_xodr(road, lane, float(s))
            if wp is None:
                continue
            if (car is None or car.get_speed_limit() == limit) and not reset(wp.transform, limit):
                unprimed += 1
                break
            if place(wp.transform) == limit:
                rows.append([road, lane, float(s), limit])
                break
    if n % 100 == 0:
        print(f"{n}/{len(signs)} signs, {len(rows)} rows", flush=True)

data = json.load(open(out)) if os.path.exists(out) else {}
data[town] = rows
open(out, "w").write(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
print(town, len(rows), "rows, lanes without a primer:", unprimed, flush=True)
car.destroy(); hero.destroy(); world.tick()
