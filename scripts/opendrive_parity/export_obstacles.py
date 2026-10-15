# Exports the footprints of CARLA's static obstacles on or next to the drivable lanes of a town (anything a vehicle
# collides with: guard rails, poles, walls, props, ...): horizontal rays at vehicle heights across every candidate
# object give solid intervals (x0, y0, x1, y1, ground z, object name, CARLA label, height above ground) in the
# OpenDRIVE frame, plus <out>.boxes.json with the CARLA bounding box of every solid object. Python 3.12 + carla wheel,
# no numpy.
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/export_obstacles.py Town04 <out.json> [port]
# Then: python scripts/opendrive_parity/static_objects_to_json.py <out_dir>  (py123d venv)
import json, math, sys
import carla

town, out = sys.argv[1], sys.argv[2]
client = carla.Client("localhost", int(sys.argv[3]) if len(sys.argv) > 3 else 2000); client.set_timeout(300)
world = client.get_world()
if not world.get_map().name.endswith(town):
    world = client.load_world(town)
settings = world.get_settings(); settings.synchronous_mode = True; settings.no_rendering_mode = True
settings.fixed_delta_seconds = 0.05; world.apply_settings(settings); world.tick()
cmap = world.get_map()

L = carla.CityObjectLabel
SOLID = {L.Buildings, L.Fences, L.Other, L.Poles, L.Vegetation, L.Walls, L.TrafficSigns, L.TrafficLight, L.Static,
         L.Dynamic, L.GuardRail, L.Bridge, L.Car, L.Truck, L.Bus, L.Motorcycle, L.Bicycle}
GROUND = {L.Roads, L.RoadLines, L.Sidewalks, L.Terrain, L.Ground, L.Bridge, L.Water}
LT = carla.LaneType
DRIVABLE = LT.Driving | LT.Shoulder | LT.Parking | LT.Border | LT.Bidirectional | LT.Restricted | LT.Biking
HEIGHTS, STEP, MARGIN, NEAR, LONG = (0.3, 0.8, 1.4), 0.1, 0.5, 1.0, 0.6  # [m]


def near_road(x, y, z):
    wp = cmap.get_waypoint(carla.Location(x, y, z), project_to_road=True, lane_type=DRIVABLE)
    if wp is None:
        return False
    return wp.transform.location.distance(carla.Location(x, y, wp.transform.location.z)) < wp.lane_width / 2 + NEAR


def ground_z(x, y, top, bottom):
    hits = world.cast_ray(carla.Location(x, y, top + 1.0), carla.Location(x, y, bottom - 3.0))
    # NOTE: Objects may reach below the road (tunnel entrances): the ground is the highest surface near their bottom.
    zs = [h.location.z for h in hits if h.label in GROUND and h.location.z <= bottom + 2.0]
    return max(zs) if zs else None


def entries(a, b):
    # Solid surfaces entered on the way from a to b (a ray reports the first hit of each component).
    return sorted(round(a.distance(h.location), 3) for h in world.cast_ray(a, b) if h.label in SOLID)


def inside(p):
    # A ray starting inside a solid hits it at once.
    return any(h.label in SOLID for h in world.cast_ray(p, carla.Location(p.x + 0.01, p.y, p.z)))


rows = []
objects = [o for o in world.get_environment_objects(L.Any) if o.type in SOLID and not o.name.startswith("Road_")]
for n, obj in enumerate(objects):
    box = obj.bounding_box
    verts = box.get_world_vertices(carla.Transform())
    bottom, top = min(v.z for v in verts), max(v.z for v in verts)
    c = box.location
    yaw = math.radians(box.rotation.yaw)
    ux, uy = (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))
    ex, ey = box.extent.x, box.extent.y
    if ex < ey:  # rows cross the short side
        ux, uy, ex, ey = uy, ux, ey, ex
    samples = [(i / 2, j / 2) for i in (-2, -1, 0, 1, 2) for j in (-2, 0, 2)]
    corner = lambda i, j: (c.x + ux[0] * ex * i + uy[0] * ey * j, c.y + ux[1] * ex * i + uy[1] * ey * j)
    if not any(near_road(*corner(i, j), bottom) for i, j in samples):
        continue
    for k in range(int(2 * (ex + MARGIN) / STEP) + 1):
        s = -ex - MARGIN + k * STEP
        mx, my = c.x + ux[0] * s, c.y + ux[1] * s
        if not near_road(mx, my, bottom):
            continue
        z0 = ground_z(mx, my, top, bottom)
        if z0 is None or bottom > z0 + HEIGHTS[-1] or top < z0 + HEIGHTS[0]:
            continue
        for h in HEIGHTS:
            a = carla.Location(mx - uy[0] * (ey + MARGIN), my - uy[1] * (ey + MARGIN), z0 + h)
            b = carla.Location(mx + uy[0] * (ey + MARGIN), my + uy[1] * (ey + MARGIN), z0 + h)
            span = a.distance(b)
            at = lambda t: carla.Location(a.x + (b.x - a.x) * t / span, a.y + (b.y - a.y) * t / span, a.z)
            outs = [span - t for t in entries(b, a)]
            for f in entries(a, b):
                e = min([t for t in outs if t >= f], default=f + STEP)
                # NOTE: A component crossed twice (pillars of one mesh) gives one long interval: keep its solid parts.
                parts = [(f, e)] if e - f <= LONG else [
                    (t, t + STEP)
                    for t in (f + k * STEP for k in range(int((e - f) / STEP) + 1))
                    if inside(at(t + STEP / 2))
                ]
                for f0, e0 in parts:
                    p, q = at(f0), at(min(e0, e))
                    rows.append([round(p.x, 3), round(-p.y, 3), round(q.x, 3), round(-q.y, 3), round(z0, 2),
                                 obj.name, str(obj.type), round(top - z0, 2)])
    if n % 500 == 0:
        print(f"{town} {n}/{len(objects)} objects, {len(rows)} intervals", flush=True)
json.dump(rows, open(out, "w"))
# CARLA bounding box of every solid object (footprint corners, label, bottom and top z): a ray also hits the neighbours
# of the object it crosses, the boxes tell them apart.
boxes = {}
for o in objects:
    vs = o.bounding_box.get_world_vertices(carla.Transform())
    corners = [[round(v.x, 3), round(-v.y, 3)] for v in vs[::2]]
    boxes[o.name] = [corners, str(o.type), round(min(v.z for v in vs), 2), round(max(v.z for v in vs), 2)]
json.dump(boxes, open(out.replace(".json", ".boxes.json"), "w"))
print(town, len(rows), "intervals")
