import numpy as np
import shapely

from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.utils.lane_helper import LANE_TYPES
from py123d.parser.opendrive.xodr_parser.lane import section_grid

MEDIAN_TYPES = {"shoulder", "median", "none", "border"}
RAISED_MEDIAN_TYPES = {"median"}
MEDIAN_MIN_WIDTH = 1.0  # [m]


def collect_lanes(roads):
    # Sample every lane of every section on a shared s-grid, oriented along the lane's driving direction.
    lanes, sampled, raised, curb_lanes = {}, {}, set(), set()
    for road_id, road in roads.items():
        for k, section in enumerate(road["sections"]):
            grid, borders = section_grid(road, k)
            sampled[road_id, k] = grid, borders
            road_type = [t for t in road["types"] if t[0] <= section["s"]]
            for i, lane in section["lanes"].items():
                if i == 0:
                    continue
                inner, outer = borders[i - np.sign(i)], borders[i]
                if i > 0:
                    inner, outer = inner[::-1], outer[::-1]
                lanes[lane_key(road_id, k, i)] = {
                    "road": road_id,
                    "section": k,
                    "id": i,
                    "type": lane["type"],
                    "inner": inner,
                    "outer": outer,
                    "road_type": road_type[-1][1] if road_type else None,
                    "speed": road_type[-1][2] if road_type else None,
                    "junction": road["junction"],
                }
            # NOTE: Uncurbed non-driving lanes between the reference line and the first driving lane (at least
            # MEDIAN_MIN_WIDTH wide) are raised medians / roundabout islands.
            for sign in (1, -1):
                ids = sorted((i for i in section["lanes"] if i * sign > 0), key=abs)
                driving = [n for n, i in enumerate(ids) if section["lanes"][i]["type"] == "driving"]
                run = ids[: driving[0]] if driving and road["junction"] == -1 else []
                is_median = run and all(
                    section["lanes"][i]["type"] in MEDIAN_TYPES and all(m["type"] != "curb" for m in section["lanes"][i]["marks"])
                    for i in run
                )
                if is_median and np.max(np.linalg.norm(borders[run[-1]] - borders[0], axis=1)) >= MEDIAN_MIN_WIDTH:
                    # NOTE: With a curb on the reference line (island kerb), the whole run up to its last median / none
                    # lane is the island.
                    island = [n for n, i in enumerate(run) if section["lanes"][i]["type"] in RAISED_MEDIAN_TYPES]
                    curbed_reference = any(m["type"] == "curb" for m in section["lanes"].get(0, {"marks": []})["marks"])
                    raised |= {
                        lane_key(road_id, k, i)
                        for n, i in enumerate(run)
                        if (curbed_reference and island and n <= island[-1]) or section["lanes"][i]["type"] in RAISED_MEDIAN_TYPES
                    }
            # NOTE: Lanes carrying a "curb" mark are the curb itself and "median" lanes are raised (CARLA renders both
            # off-road).
            raised |= {
                lane_key(road_id, k, i)
                for i, lane in section["lanes"].items()
                if i and (lane["type"] == "median" or any(m["type"] == "curb" for m in lane["marks"]))
            }
            curb_lanes |= {
                lane_key(road_id, k, i)
                for i, lane in section["lanes"].items()
                if i and lane["type"] != "median" and any(m["type"] == "curb" for m in lane["marks"])
            }

    # NOTE: Non-driving lanes overlapping a raised median / island (e.g. junction shoulders) are raised as well. Curb
    # lanes are thin kerbs: they neither spread raising nor keep holes open.
    polygon_of = lambda ln: shapely.Polygon(np.vstack([ln["inner"], ln["outer"][::-1]])[:, :2]).buffer(0)
    raised_area = shapely.union_all([polygon_of(lanes[k]) for k in raised - curb_lanes])
    raised |= {
        k
        for k, ln in lanes.items()
        if ln["type"] in MEDIAN_TYPES
        and len(ln["inner"]) > 1
        and polygon_of(ln).area > 0
        and polygon_of(ln).intersection(raised_area).area > 0.1 * polygon_of(ln).area
    }
    return lanes, sampled, raised, curb_lanes


def lane_successors(roads, junctions, road_id, section_idx, lane_id):
    # Port of libcarla MapBuilder::GetLaneNext / GetJunctionLanes (RHT: lanes with id <= 0 follow the road direction).
    road = roads[road_id]
    sections = road["sections"]
    lane = sections[section_idx]["lanes"][lane_id]
    forward = lane_id <= 0
    next_road, next_lane = (road["succ"], lane["succ"]) if forward else (road["pred"], lane["pred"])
    s = sections[section_idx]["s"]

    def edge_lane(r, i):
        candidates = [k for k, sec in enumerate(roads[r]["sections"]) if i in sec["lanes"]]
        return [(r, candidates[0] if i <= 0 else candidates[-1], i)] if candidates else []

    if (not forward and s > 0) or (forward and any(sec["s"] > s for sec in sections)):
        if next_lane == 0:
            return []
        ks = [k for k, sec in enumerate(sections) if (sec["s"] > s if forward else sec["s"] < s)]
        ks = [k for k in (ks if forward else ks[::-1]) if next_lane in sections[k]["lanes"]]
        return [(road_id, ks[0], next_lane)] if ks else []
    if next_road in roads:
        return edge_lane(next_road, next_lane) if next_lane != 0 else []
    if next_road not in junctions:
        return []

    result = []
    for connection in junctions[next_road].findall("connection"):
        conn = roads[int(connection.get("connectingRoad"))]
        if conn["pred"] == road_id:
            result += [x for i, ln in conn["sections"][0]["lanes"].items() if ln["pred"] == lane_id for x in edge_lane(conn["id"], i)]
        if conn["succ"] == road_id:
            result += [x for i, ln in conn["sections"][-1]["lanes"].items() if ln["succ"] == lane_id for x in edge_lane(conn["id"], i)]
    return result


def collect_topology(roads, junctions, lanes):
    # Topology (libcarla semantics) of all lanes, py123d lanes only link to py123d lanes.
    all_successors = {
        key: list(dict.fromkeys(lane_key(*x) for x in lane_successors(roads, junctions, ln["road"], ln["section"], ln["id"])))
        for key, ln in lanes.items()
    }
    all_predecessors = {key: [] for key in lanes}
    for key, nexts in all_successors.items():
        for nxt in nexts:
            all_predecessors[nxt].append(key)
    is_lane = {key for key, ln in lanes.items() if ln["type"] in LANE_TYPES}
    predecessors = {key: [p for p in all_predecessors[key] if p in is_lane] for key in is_lane}
    successors = {key: [] for key in is_lane}
    for key, preds in predecessors.items():
        for p in preds:
            successors[p].append(key)
    return all_successors, all_predecessors, is_lane, predecessors, successors
