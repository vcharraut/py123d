# Exports the stop and yield signs of a CARLA town to the OpenDRIVE frame: [kind, x, y, z of the trigger volume centre,
# [(road id, lane id, s, first, last, road id, lane id, s)]], one entry per approach lane that reaches the sign: the
# lane's last position before the junction, the distances before that junction entry of the first and last lane centre
# positions at which scenario_runner's RunningStopTest counts a vehicle as affected by the sign (negative: inside the
# junction), and the last of these positions. Affected: the vehicle's waypoint within PROXIMITY of the trigger centre
# and one of its waypoints up to PROXIMITY ahead inside the trigger volume (x 1.2, axis-aligned). A stop only counts
# there. Python 3.12 + carla wheel.
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/export_stop_signs.py <out.json> <port> Town03 [Town04 ...]
import json, os, sys
import carla

PROXIMITY, STEP, BOX_SCALE, WALK, HORIZON = 4.0, 0.5, 1.2, 0.1, 60.0  # [m], [m], [-], [m], [m]
out, port, towns = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
client = carla.Client("localhost", port); client.set_timeout(300)
signs = json.load(open(out)) if os.path.exists(out) else {}
for town in towns:
    world = client.get_world()
    if not world.get_map().name.endswith(town):
        world = client.load_world(town)
    cmap = world.get_map()

    def affected(wp, centre, extent):
        if wp.transform.location.distance(centre) > PROXIMITY:
            return False
        for _ in range(int(PROXIMITY / STEP) + 1):
            loc = wp.transform.location
            if abs(loc.x - centre.x) <= extent.x * BOX_SCALE and abs(loc.y - centre.y) <= extent.y * BOX_SCALE:
                return True
            nxt = wp.next(STEP)
            if not nxt:
                return False
            wp = nxt[0]
        return False

    def to_entry(wp):
        # Last waypoint before the junction ahead and the distance to it (negative from inside the junction).
        d = 0.0
        while wp.is_junction and d > -HORIZON:
            prev = wp.previous(WALK)
            if not prev:
                return None, None
            wp, d = prev[0], d - WALK
        while d < HORIZON:
            nxt = wp.next(WALK)
            if not nxt:
                return None, None
            if nxt[0].is_junction:
                return wp, d
            wp, d = nxt[0], d + WALK
        return None, None

    rows = []
    for actor in world.get_actors().filter("traffic.*"):
        kind = {"traffic.stop": "stop", "traffic.yield": "yield"}.get(actor.type_id)
        if kind is None:
            continue
        centre = actor.get_transform().transform(actor.trigger_volume.location)
        extent = actor.trigger_volume.extent
        # NOTE: Lanes reaching the sign: the driving lanes under the (enlarged) trigger volume.
        lanes = {}
        for i in range(-8, 9):
            for j in range(-8, 9):
                dx, dy = extent.x * BOX_SCALE * i / 8, extent.y * BOX_SCALE * j / 8
                p = carla.Location(centre.x + dx, centre.y + dy, centre.z)
                wp = cmap.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is not None and wp.transform.location.distance(p) < wp.lane_width / 2:
                    lanes.setdefault((wp.road_id, wp.lane_id), wp)
        approaches = {}
        for wp in lanes.values():
            # Walk the lane centre from 8 m upstream of the trigger to 12 m past it, in driving direction.
            back = wp.previous(8.0)
            wp = back[0] if back else wp
            hits = []
            for _ in range(int(20.0 / WALK)):
                if affected(wp, centre, extent):
                    hits.append(wp)
                nxt = wp.next(WALK)
                if not nxt:
                    break
                wp = nxt[0]
            if not hits:
                continue
            (entry, d_first), (_, d_last) = to_entry(hits[0]), to_entry(hits[-1])
            if entry is None:
                continue
            key = (entry.road_id, entry.lane_id)
            _, first, last, end = approaches.get(key, (None, d_first, d_last, hits[-1]))
            approaches[key] = (entry, max(first, d_first), min(last, d_last), end if last < d_last else hits[-1])
        rows.append([kind, round(centre.x, 2), round(-centre.y, 2), round(centre.z, 2), [
            [e.road_id, e.lane_id, round(e.s, 2), round(f, 2), round(l, 2), w.road_id, w.lane_id, round(w.s, 2)]
            for e, f, l, w in approaches.values()
        ]])
    signs[town] = rows
    print(town, len(rows), "signs,", sum(len(r[4]) for r in rows), "approach lanes", flush=True)
json.dump(signs, open(out, "w"))
