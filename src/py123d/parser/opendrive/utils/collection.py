import xml.etree.ElementTree as ET
from typing import Any, Dict, Final, List, Set, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.utils.lane_helper import LANE_TYPES
from py123d.parser.opendrive.xodr_parser.lane import section_grid

_MEDIAN_TYPES: Final[Set[str]] = {"shoulder", "median", "none", "border"}
_RAISED_MEDIAN_TYPES: Final[Set[str]] = {"median"}
_MEDIAN_MIN_WIDTH: Final[float] = 1.0  # [m]
KERB_MAX_WIDTH: Final[float] = 1.0  # [m]


def collect_lanes(
    roads: Dict[int, Dict[str, Any]],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[Tuple[int, int], Tuple[npt.NDArray[np.float64], Dict[int, npt.NDArray[np.float64]]]],
    Set[str],
    Set[str],
]:
    """Samples every lane of every section on a shared s-grid, oriented along the lane's driving direction.

    :param roads: Parsed roads, keyed by id.
    :return: Lanes keyed by lane id, the s-grid and lane borders per (road id, section index), the ids of raised
        lanes (medians, islands, curbs) and the ids of curb lanes.
    """
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
            # _MEDIAN_MIN_WIDTH wide) are raised medians / roundabout islands.
            for sign in (1, -1):
                ids = sorted((i for i in section["lanes"] if i * sign > 0), key=abs)
                driving = [n for n, i in enumerate(ids) if section["lanes"][i]["type"] == "driving"]
                run = ids[: driving[0]] if driving and road["junction"] == -1 else []
                is_median = run and all(
                    section["lanes"][i]["type"] in _MEDIAN_TYPES
                    and all(m["type"] != "curb" for m in section["lanes"][i]["marks"])
                    for i in run
                )
                if is_median and np.max(np.linalg.norm(borders[run[-1]] - borders[0], axis=1)) >= _MEDIAN_MIN_WIDTH:
                    # NOTE: With a curb on the reference line (island kerb), the whole run up to its last median / none
                    # lane is the island.
                    island = [n for n, i in enumerate(run) if section["lanes"][i]["type"] in _RAISED_MEDIAN_TYPES]
                    reference_marks = section["lanes"].get(0, {"marks": []})["marks"]
                    curbed_reference = any(m["type"] == "curb" for m in reference_marks)
                    raised |= {
                        lane_key(road_id, k, i)
                        for n, i in enumerate(run)
                        if (curbed_reference and island and n <= island[-1])
                        or section["lanes"][i]["type"] in _RAISED_MEDIAN_TYPES
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
            # NOTE: A "curb" mark lies on the kerb / sidewalk border, whichever lane carries it (its outer border, lane
            # 0: the reference line). Narrow non-driving strips on either side of that border are the kerb as well.
            for i, lane in section["lanes"].items():
                if not any(m["type"] == "curb" for m in lane["marks"]):
                    continue
                for j in (i, i + np.sign(i)) if i else (1, -1):
                    kerb = section["lanes"].get(j, {"type": "driving"})["type"] not in ("driving", "sidewalk", "median")
                    width = np.max(np.linalg.norm(borders[j] - borders[j - np.sign(j)], axis=1)) if kerb else np.inf
                    if width < KERB_MAX_WIDTH:
                        raised.add(lane_key(road_id, k, j))
                        curb_lanes.add(lane_key(road_id, k, j))

    # NOTE: Non-driving lanes overlapping a raised median / island (e.g. junction shoulders) are raised as well. Curb
    # lanes are thin kerbs: they neither spread raising nor keep holes open.
    polygon_of = lambda lane: shapely.Polygon(np.vstack([lane["inner"], lane["outer"][::-1]])[:, :2]).buffer(0)
    raised_area = shapely.union_all([polygon_of(lanes[k]) for k in lanes if k in raised and k not in curb_lanes])
    raised |= {
        k
        for k, lane in lanes.items()
        if lane["type"] in _MEDIAN_TYPES
        and len(lane["inner"]) > 1
        and polygon_of(lane).area > 0
        and polygon_of(lane).intersection(raised_area).area > 0.1 * polygon_of(lane).area
    }
    return lanes, sampled, raised, curb_lanes


def _lane_successors(
    roads: Dict[int, Dict[str, Any]], junctions: Dict[int, ET.Element], road_id: int, section_idx: int, lane_id: int
) -> List[Tuple[int, int, int]]:
    # Port of libcarla MapBuilder::GetLaneNext / GetJunctionLanes (RHT: lanes with id <= 0 follow the road direction).
    road = roads[road_id]
    sections = road["sections"]
    lane = sections[section_idx]["lanes"][lane_id]
    forward = lane_id <= 0
    next_road, next_lane = (road["succ"], lane["succ"]) if forward else (road["pred"], lane["pred"])
    s = sections[section_idx]["s"]

    def edge_lane(r: int, i: int) -> List[Tuple[int, int, int]]:
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
            result += [
                x
                for i, conn_lane in conn["sections"][0]["lanes"].items()
                if conn_lane["pred"] == lane_id
                for x in edge_lane(conn["id"], i)
            ]
        if conn["succ"] == road_id:
            result += [
                x
                for i, conn_lane in conn["sections"][-1]["lanes"].items()
                if conn_lane["succ"] == lane_id
                for x in edge_lane(conn["id"], i)
            ]
    return result


def collect_topology(
    roads: Dict[int, Dict[str, Any]], junctions: Dict[int, ET.Element], lanes: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, None], Dict[str, List[str]], Dict[str, List[str]]]:
    """Computes the lane topology (libcarla semantics), py123d lanes only link to py123d lanes.

    :param roads: Parsed roads, keyed by id.
    :param junctions: Raw junction elements, keyed by id.
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :return: Successors and predecessors over all lanes, the ordered ids of py123d lanes (driving and biking), and
        successors and predecessors restricted to py123d lanes.
    """
    all_successors = {
        key: list(
            dict.fromkeys(
                lane_key(*x) for x in _lane_successors(roads, junctions, lane["road"], lane["section"], lane["id"])
            )
        )
        for key, lane in lanes.items()
    }
    all_predecessors = {key: [] for key in lanes}
    for key, nexts in all_successors.items():
        for nxt in nexts:
            all_predecessors[nxt].append(key)
    is_lane = dict.fromkeys(key for key, lane in lanes.items() if lane["type"] in LANE_TYPES)
    predecessors = {key: [p for p in all_predecessors[key] if p in is_lane] for key in is_lane}
    successors = {key: [] for key in is_lane}
    for key, preds in predecessors.items():
        for p in preds:
            successors[p].append(key)
    return all_successors, all_predecessors, is_lane, predecessors, successors
