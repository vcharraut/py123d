# Exports the ground surface around every junction of a CARLA town: vertical rays on a COARSE grid over the junction's
# driving lanes plus MARGIN give, per cell, whether the topmost ground near the road height is paved (labelled Roads or
# RoadLines) and its height; cells at a change of surface or height are cast again on SUB x SUB rays. Writes one entry
# per junction in the OpenDRIVE frame: grid origin and size, the coarse cells, the refined cells and the junction lane
# positions. Python 3.12 + carla wheel, no numpy.
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/export_junction_pavement.py Town07 <out_dir>/Town07.json [port]
# Then: python scripts/opendrive_parity/junction_pavement_to_json.py <out_dir>  (py123d venv)
import json, math, sys
import carla

town, out = sys.argv[1], sys.argv[2]
client = carla.Client("localhost", int(sys.argv[3]) if len(sys.argv) > 3 else 2000); client.set_timeout(300)
world = client.get_world()
if not world.get_map().name.endswith(town):
    world = client.load_world(town)
cmap = world.get_map()

L = carla.CityObjectLabel
PAVED = {L.Roads, L.RoadLines}
# NOTE: The surface is the topmost ground hit: guard rails, poles and props stand on it, and hits without a label are
# invisible volumes (a flat box lies over every junction, trigger boxes over the lanes).
GROUND = PAVED | {L.Sidewalks, L.Terrain, L.Ground, L.Bridge, L.Water, L.RailTrack}
COARSE, SUB, MARGIN, REACH, STEP_Z = 0.25, 5, 5.0, 1.5, 0.04  # [m], [-], [m], [m], [m]


def surface(x, y, z):
    # (paved, height) of the topmost surface within REACH of the road height z, (None, None) without one.
    # NOTE: A ray starting inside a volume hits it at once: those hits are no surface either.
    hits = world.cast_ray(carla.Location(x, -y, z + REACH), carla.Location(x, -y, z - REACH))
    hit = next((h for h in hits if h.label in GROUND and h.location.z < z + REACH - 0.01), None)
    return (None, None) if hit is None else (hit.label in PAVED, round(hit.location.z, 3))


junctions = {}
for wp in cmap.generate_waypoints(2.0):
    if wp.is_junction:
        junctions.setdefault(wp.junction_id, wp.get_junction())

rows = []
for n, (junction_id, junction) in enumerate(sorted(junctions.items())):
    seeds = []
    for start, _ in junction.get_waypoints(carla.LaneType.Driving):
        wp = start
        for _ in range(1000):
            if wp is None or not wp.is_junction or wp.junction_id != junction_id:
                break
            seeds.append(wp.transform.location)
            nexts = wp.next(1.0)
            wp = nexts[0] if nexts else None
    if not seeds:
        continue
    x0, y0 = min(s.x for s in seeds) - MARGIN, min(-s.y for s in seeds) - MARGIN
    nx = math.ceil((max(s.x for s in seeds) + MARGIN - x0) / COARSE)
    ny = math.ceil((max(-s.y for s in seeds) + MARGIN - y0) / COARSE)
    z_mean = sum(s.z for s in seeds) / len(seeds)

    paved, height, road_z = [], [], []
    for j in range(ny):
        for i in range(nx):
            x, y = x0 + (i + 0.5) * COARSE, y0 + (j + 0.5) * COARSE
            # NOTE: The road height here is the one of the nearest lane (junctions on slopes, roads above others).
            z = cmap.get_waypoint(carla.Location(x, -y, z_mean), lane_type=carla.LaneType.Any).transform.location.z
            p, h = surface(x, y, z)
            paved.append(bool(p)); height.append(h); road_z.append(z)

    at = lambda i, j: j * nx + i
    inside = lambda i, j: 0 <= i < nx and 0 <= j < ny
    changes = set()
    for j in range(ny):
        for i in range(nx):
            k = at(i, j)
            around = [(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di or dj) and inside(i + di, j + dj)]
            step = lambda a, b: paved[a] and paved[b] and abs(height[a] - height[b]) > STEP_Z
            if any(paved[at(*c)] != paved[k] or step(k, at(*c)) for c in around):
                changes.update([(i, j), *around])
    fine = {}
    for i, j in sorted(changes):
        cells = [
            surface(x0 + (i + (a + 0.5) / SUB) * COARSE, y0 + (j + (b + 0.5) / SUB) * COARSE, road_z[at(i, j)])
            for b in range(SUB) for a in range(SUB)
        ]
        fine[f"{i},{j}"] = ["".join("1" if p else "0" for p, _ in cells), [h for _, h in cells]]
    rows.append({
        "id": junction_id, "x0": x0, "y0": y0, "nx": nx, "ny": ny, "step": COARSE, "sub": SUB,
        "paved": "".join("1" if p else "0" for p in paved), "z": height, "fine": fine,
        "seeds": [[round(s.x, 2), round(-s.y, 2), round(s.z, 2)] for s in seeds],
    })
    print(f"{town} junction {n + 1}/{len(junctions)}: {nx * ny} cells, {len(fine)} refined", flush=True)
json.dump(rows, open(out, "w"))
print(town, len(rows), "junctions")
