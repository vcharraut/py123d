from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import shapely
from scipy.spatial import cKDTree

from py123d.datatypes import (
    BaseMapObject,
    Carpark,
    Crosswalk,
    GenericDrivable,
    Intersection,
    IntersectionType,
    Lane,
    LaneGroup,
    LaneType,
    RoadEdge,
    RoadEdgeType,
    RoadLine,
    RoadLineType,
    StopZone,
    StopZoneType,
    Walkway,
)
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.geometry.polyline import Polyline3D
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import split_line_geometry_by_max_length
from py123d.parser.opendrive.xodr import eval_borders, eval_poly, eval_reference, load_xodr, section_at, section_grid

LANE_TYPES = {"driving": LaneType.SURFACE_STREET, "biking": LaneType.BIKE_LANE}
SURFACE_TYPES = {
    "parking": Carpark,
    "shoulder": GenericDrivable,
    "border": GenericDrivable,
    "bidirectional": GenericDrivable,
    "restricted": GenericDrivable,
    "sidewalk": Walkway,
}
ROAD_LINE_TYPES = {
    ("solid", "white"): RoadLineType.SOLID_WHITE,
    ("solid", "yellow"): RoadLineType.SOLID_YELLOW,
    ("broken", "white"): RoadLineType.DASHED_WHITE,
    ("broken", "yellow"): RoadLineType.DASHED_YELLOW,
    ("solid solid", "white"): RoadLineType.DOUBLE_SOLID_WHITE,
    ("solid solid", "yellow"): RoadLineType.DOUBLE_SOLID_YELLOW,
    ("broken broken", "white"): RoadLineType.DOUBLE_DASH_WHITE,
    ("broken broken", "yellow"): RoadLineType.DOUBLE_DASH_YELLOW,
    ("solid broken", "white"): RoadLineType.SOLID_DASH_WHITE,
    ("solid broken", "yellow"): RoadLineType.SOLID_DASH_YELLOW,
    ("broken solid", "white"): RoadLineType.DASH_SOLID_WHITE,
    ("broken solid", "yellow"): RoadLineType.DASH_SOLID_YELLOW,
}
KEEP_HOLE_RADIUS = 12.0  # [m], larger holes in the road surface are off-road areas (blocks, roundabouts)
ISLAND_CURB_FRACTION = 0.05  # holes with this much curb on their boundary are raised islands ...
ISLAND_MIN_RADIUS = 1.0  # [m] ... unless they are thin slivers
LAYER_DZ = 1.0  # [m], touching surfaces with larger height difference are on different layers (bridges)
MAX_ROAD_EDGE_LENGTH = 100.0  # [m]
STOP_LINE_DEPTH = 0.5  # [m]
STOP_LINE_SEARCH = 15.0  # [m], upstream distance searched for painted stop lines / crosswalks
STOP_LINE_JUNCTION_SEARCH = 20.0  # [m], painted stop lines inside junctions lie close to the junction entry
STOP_LINE_MAX_TURN = 30.0  # [deg], ... before the junction lane turns
STOP_LINE_SNAP = 0.2  # [m]
STOP_LINE_MIN_ANGLE = 60.0  # [deg], painted stop lines must cross the lane at least this steeply
MEDIAN_TYPES = {"shoulder", "median", "none", "border"}
RAISED_MEDIAN_TYPES = {"median"}
MEDIAN_MIN_WIDTH = 1.0  # [m]
CLOSING = 0.05  # [m], closes slits between adjacent lanes
GAP_CLOSING = 0.5  # [m]
MIN_SURFACE_AREA = 10.0  # [m^2], smaller disconnected road surface pieces are dropped (and thin ones without driving lanes)
STOP_ZONE_TYPES = {"1000001": StopZoneType.TRAFFIC_LIGHT, "206": StopZoneType.STOP_SIGN, "205": StopZoneType.YIELD_SIGN}


class OpenDriveMapParser(BaseMapParser):
    """Lightweight, picklable handle to one OpenDRIVE map."""

    def __init__(self, xodr_path: Path, location: Optional[str] = None) -> None:
        self._xodr_path = xodr_path
        self._location = location

    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata for this OpenDRIVE map."""
        _location = self._location or self._xodr_path.name.removesuffix("".join(self._xodr_path.suffixes))
        return MapMetadata(dataset="opendrive", location=_location, map_has_z=True, map_is_per_log=False)

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily from the XODR file."""
        yield from iter_xodr_map_objects(self._xodr_path)


def lane_key(road_id, section_idx, lane_id):
    return f"{road_id}_{section_idx}_{'left' if lane_id > 0 else 'right'}_{lane_id}"


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


def iter_xodr_map_objects(xodr_file: Path) -> Iterator[BaseMapObject]:
    roads, junctions = load_xodr(xodr_file)

    # 1. Sample every lane of every section on a shared s-grid, oriented along the lane's driving direction.
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

    # 2. Topology (libcarla semantics) of all lanes, py123d lanes only link to py123d lanes.
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

    # 3. Lanes and lane groups: one group per road, section and side, lanes ordered inner to outer.
    groups = {}
    for key in is_lane:
        groups.setdefault(key.rsplit("_", 1)[0], []).append(key)
    lane_group_of = {key: group_id for group_id, keys in groups.items() for key in keys}

    for group_id, keys in groups.items():
        keys.sort(key=lambda k: abs(lanes[k]["id"]))
        for key in keys:
            ln = lanes[key]
            neighbour = lambda i: lane_key(ln["road"], ln["section"], i) if lane_key(ln["road"], ln["section"], i) in is_lane else None
            step = np.sign(ln["id"])
            yield Lane(
                object_id=key,
                lane_type=LaneType.FREEWAY if ln["road_type"] == "motorway" and ln["type"] == "driving" else LANE_TYPES[ln["type"]],
                left_boundary=Polyline3D.from_array(ln["inner"]),
                right_boundary=Polyline3D.from_array(ln["outer"]),
                centerline=Polyline3D.from_array(0.5 * (ln["inner"] + ln["outer"])),
                lane_group_id=group_id,
                left_lane_id=neighbour(ln["id"] - step),
                right_lane_id=neighbour(ln["id"] + step),
                predecessor_ids=predecessors[key],
                successor_ids=successors[key],
                speed_limit_mps=ln["speed"],
            )
        first = lanes[keys[0]]
        yield LaneGroup(
            object_id=group_id,
            lane_ids=keys,
            left_boundary=Polyline3D.from_array(first["inner"]),
            right_boundary=Polyline3D.from_array(lanes[keys[-1]]["outer"]),
            intersection_id=first["junction"] if first["junction"] != -1 else None,
            predecessor_ids=list(dict.fromkeys(lane_group_of[p] for k in keys for p in predecessors[k])),
            successor_ids=list(dict.fromkeys(lane_group_of[s] for k in keys for s in successors[k])),
        )

    # 5. Road lines from road marks, each mark lies on the outer border of its lane (lane 0: reference line).
    curbs = []
    for (road_id, k), (grid, borders) in sampled.items():
        road, sections = roads[road_id], roads[road_id]["sections"]
        s1 = sections[k + 1]["s"] if k + 1 < len(sections) else road["length"]
        for i, lane in sections[k]["lanes"].items():
            marks = lane["marks"]
            for j, mark in enumerate(marks):
                a, b = mark["s"], min(marks[j + 1]["s"] if j + 1 < len(marks) else s1, s1)
                # NOTE: Marks between two raised lanes are inside a median / island.
                inside_raised = {lane_key(road_id, k, i), lane_key(road_id, k, i + np.sign(i))} <= raised
                if mark["type"] == "none" or b - a < 1e-3 or inside_raised:
                    continue
                ends = eval_borders(road, k, np.array([a, b]))[i]
                points = np.vstack([ends[:1], borders[i][(grid > a) & (grid < b)], ends[1:]])
                if mark["type"] == "curb":
                    curbs.append(shapely.LineString(points[:, :2]))
                    continue
                color = "white" if mark["color"] in (None, "standard") else mark["color"]
                yield RoadLine(
                    object_id=f"{road_id}_{k}_{i}_mark_{j}",
                    road_line_type=ROAD_LINE_TYPES[mark["type"], color],
                    polyline=Polyline3D.from_array(points),
                )

    # 6. Crosswalks (libcarla Map::GetAllCrosswalkZones): outline corners relative to the offset reference line.
    # Painted "StopLine*" objects use the same outline convention and are kept to place stop zones.
    crosswalks, painted_stop_lines, painted_directions = [], [], []
    for road_id, road in roads.items():
        for obj in road["objects"].findall("object") if road["objects"] is not None else []:
            is_crosswalk = obj.get("type") == "crosswalk" or "crosswalk" in obj.get("name", "").lower()
            if not is_crosswalk and not (obj.get("name", "").startswith("StopLine") and obj.find("outline") is not None):
                continue
            corners = np.array([[float(c.get(a)) for a in "uvz"] for c in obj.find("outline").findall("cornerLocal")])
            s, t = np.array([float(obj.get("s"))]), float(obj.get("t", 0.0))
            base = eval_borders(road, section_at(road, s[0]), s)[0][0]
            # NOTE: libcarla's lane 0 transform: heading plus the raw laneOffset derivative, pitch is the raw elevation
            # derivative. The whole corner frame is rotated with that transform.
            dh = lambda records: np.diff(eval_poly(records, s[0] + np.array([-1e-6, 1e-6])))[0] / 2e-6
            h = eval_reference(road, s)[2][0] + dh(road["lane_offset"])
            a, p = h + float(obj.get("hdg", 0.0)), dh(road["elevation"])
            u_axis = np.array([np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), np.sin(p)])
            v_axis = np.array([-np.sin(a), np.cos(a), 0.0])
            z_axis = np.cross(u_axis, v_axis)
            pivot = base + t * np.array([-np.cos(p) * np.sin(h), np.cos(p) * np.cos(h), np.sin(p)])
            outline = pivot + corners @ np.stack([u_axis, v_axis, z_axis])
            if not is_crosswalk:
                painted_stop_lines.append(shapely.LineString(outline[:, :2]).buffer(STOP_LINE_SNAP))
                edges = np.diff(outline[:, :2], axis=0)
                longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
                painted_directions.append(longest / np.linalg.norm(longest))
                continue
            crosswalks.append(shapely.Polygon(outline[:, :2]) if len(outline) > 2 else shapely.Polygon())
            yield Crosswalk(object_id=f"{road_id}_object_{obj.get('id')}", outline=Polyline3D.from_array(np.vstack([outline, outline[:1]])))
    crosswalk_tree = shapely.STRtree([c if c.is_valid else shapely.make_valid(c) for c in crosswalks])
    painted_tree = shapely.STRtree(painted_stop_lines)
    painted_directions = np.array(painted_directions).reshape(-1, 2)

    # 7. Stop zones: stop lines across the lane where it enters the junction (CARLA BasicAgent stop waypoint),
    # found from the lanes each traffic light / stop / yield signal is valid for.
    signal_types, references = {}, []
    for road_id, road in roads.items():
        for el in road["signals"] if road["signals"] is not None else []:
            if el.tag == "signal":
                signal_types[el.get("id")] = el.get("type")
            s = min(max(float(el.get("s")), 0.0), road["length"] - 1e-5)
            validities = [(int(v.get("fromLane")), int(v.get("toLane"))) for v in el.findall("validity")]
            if not validities:
                ids = road["sections"][section_at(road, s)]["lanes"].keys()
                default = (((1, max(ids)), el.get("orientation") != "-"), ((min(ids), -1), el.get("orientation") != "+"))
                validities = [v for v, use in default if use and v[0] <= v[1]]
            if not validities or any(v != (0, 0) for v in validities):
                references.append((el.get("id"), road_id, s, validities))

    stop_lines = {}
    for signal_id, road_id, s, validities in references:
        zone_type = STOP_ZONE_TYPES.get(signal_types.get(signal_id))
        k = section_at(roads[road_id], s)
        for from_lane, to_lane in validities if zone_type is not None else []:
            for lane_id in range(min(from_lane, to_lane), max(from_lane, to_lane) + 1):
                key = lane_key(road_id, k, lane_id)
                if key not in lanes or lanes[key]["type"] != "driving":
                    continue
                preds = all_predecessors[key]
                at_start = lanes[key]["junction"] != -1 and len(preds) != 1
                if lanes[key]["junction"] != -1 and len(preds) == 1:
                    key = preds[0]
                for _ in range(100) if not at_start else []:
                    nexts = all_successors[key]
                    if not nexts or lanes[nexts[0]]["junction"] != -1:
                        break
                    key = nexts[0]
                stop_lines.setdefault((key, at_start), zone_type)

    def section_range(ln):
        sections, road = roads[ln["road"]]["sections"], roads[ln["road"]]
        return sections[ln["section"]]["s"], sections[ln["section"] + 1]["s"] if ln["section"] + 1 < len(sections) else road["length"]

    def lane_hits(key, from_end, tree, length=STOP_LINE_SEARCH, directions=None):
        # s positions (0.1 m apart, walking away from the lane end or start) whose lane centre lies in a tree geometry.
        # With line directions, only lines crossing the lane at more than STOP_LINE_MIN_ANGLE count.
        ln, (s0, s1) = lanes[key], section_range(lanes[key])
        forward = ln["id"] <= 0
        s_from = (s1 if forward else s0) if from_end else (s0 if forward else s1)
        s_near = np.linspace(s_from, s_from + (-1 if forward == from_end else 1) * length, int(length / 0.1) + 1)
        s_near = s_near[(s_near >= s0) & (s_near <= s1)]
        borders = eval_borders(roads[ln["road"]], ln["section"], s_near)
        if len(s_near) < 2:
            return np.zeros(0)
        centres = 0.5 * (borders[ln["id"]] + borders[ln["id"] - np.sign(ln["id"])])[:, :2]
        points, geometries = tree.query(shapely.points(centres), predicate="intersects")
        if directions is not None:
            tangents = np.gradient(centres, axis=0)
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
            crossing = np.abs(np.sum(tangents[points] * directions[geometries], axis=1)) < np.cos(np.radians(STOP_LINE_MIN_ANGLE))
            # NOTE: The lane must not have turned yet (painted lines of other approaches cross turning lanes).
            straight = np.sum(tangents[points] * tangents[0], axis=1) > np.cos(np.radians(STOP_LINE_MAX_TURN))
            points = points[crossing & straight]
        return s_near[np.unique(points)]

    def stop_zone(object_id, zone_type, key, s_stop, controlled, painted=False):
        # NOTE: The 0.5 m strip lies upstream of the stop position, across the full lane width (oriented along the
        # painted line if there is one). The controlled lanes start at the stop line (junction lanes), like
        # traffic-light lane connectors.
        ln, (s0, s1) = lanes[key], section_range(lanes[key])
        upstream = -np.sign(0.5 - ln["id"]) * STOP_LINE_DEPTH
        s_pair = np.clip(np.array([s_stop + upstream, s_stop]), s0, s1)
        borders = eval_borders(roads[ln["road"]], ln["section"], s_pair)
        inner, outer = borders[ln["id"] - np.sign(ln["id"])], borders[ln["id"]]
        if painted:
            centre = 0.5 * (inner[1] + outer[1])
            across = painted_directions[painted_tree.nearest(shapely.Point(centre[:2]))]
            across = across * np.sign(np.dot(across, outer[1, :2] - inner[1, :2]))
            back = np.array([-across[1], across[0]])
            back = back * (1.0 if np.dot(back, inner[0, :2] - inner[1, :2]) >= 0 else -1.0)
            half = 0.5 * np.linalg.norm(outer[1, :2] - inner[1, :2])
            ends = np.array([centre[:2] - half * across, centre[:2] + half * across])
            inner = np.column_stack([np.vstack([ends[0] + STOP_LINE_DEPTH * back, ends[0]]), inner[:, 2]])
            outer = np.column_stack([np.vstack([ends[1] + STOP_LINE_DEPTH * back, ends[1]]), outer[:, 2]])
        return StopZone(
            object_id=object_id,
            stop_zone_type=zone_type,
            outline=Polyline3D.from_array(np.vstack([inner, outer[::-1], inner[:1]])),
            lane_ids=controlled,
        )

    signal_zones = []
    for (key, at_start), zone_type in stop_lines.items():
        controlled = [key] if at_start else [s for s in all_successors[key] if s in is_lane] or [key]
        # NOTE: The vehicle stops at the painted stop line (on the approach or the first junction lane), else before a
        # crosswalk on the approach, else at the junction entry (CARLA BasicAgent stop waypoint).
        painted_before = lane_hits(key, True, painted_tree, directions=painted_directions) if not at_start else np.zeros(0)
        painted_after = (
            lane_hits(controlled[0], False, painted_tree, STOP_LINE_JUNCTION_SEARCH, painted_directions)
            if controlled[0] != key or at_start
            else np.zeros(0)
        )
        crossing = lane_hits(key, True, crosswalk_tree) if not at_start else np.zeros(0)
        if len(painted_before):
            stop_key, s_stop = key, np.median(painted_before)
        elif len(painted_after):
            stop_key, s_stop = controlled[0], np.median(painted_after)
        elif len(crossing):
            stop_key = key
            s_stop = crossing.min() if lanes[key]["id"] <= 0 else crossing.max()
        else:
            stop_key = controlled[0] if at_start else key
            s0, s1 = section_range(lanes[stop_key])
            s_stop = (s1 if lanes[stop_key]["id"] <= 0 else s0) if not at_start else (s0 if lanes[stop_key]["id"] <= 0 else s1)
        zone = stop_zone(
            f"stop_{key}{'_start' if at_start else ''}", zone_type, stop_key, s_stop, controlled, len(painted_before) + len(painted_after) > 0
        )
        signal_zones.append(zone.shapely_polygon)
        yield zone

    # NOTE: Painted stop lines without a signal are stop lines as well.
    signal_tree = shapely.STRtree(signal_zones)
    driving_keys = [k for k in is_lane if lanes[k]["type"] == "driving"]
    centre_tree = shapely.STRtree([shapely.LineString(0.5 * (lanes[k]["inner"] + lanes[k]["outer"])[:, :2]) for k in driving_keys])
    for n, painted in enumerate(painted_stop_lines):
        if len(signal_tree.query(painted, predicate="intersects")):
            continue
        for key in (driving_keys[j] for j in centre_tree.query(painted, predicate="intersects")):
            in_junction = lanes[key]["junction"] != -1
            length = STOP_LINE_JUNCTION_SEARCH if in_junction else STOP_LINE_SEARCH
            hits = lane_hits(key, not in_junction, shapely.STRtree([painted]), length, painted_directions[n : n + 1])
            controlled = [key] if lanes[key]["junction"] != -1 else [s for s in all_successors[key] if s in is_lane] or [key]
            if len(hits):
                yield stop_zone(
                    f"painted_{n}_{key}",
                    StopZoneType.STOP_SIGN,
                    key,
                    np.median(hits),
                    controlled,
                    painted=True,
                )

    # 8. Road surface = all lanes but sidewalks, raised medians/islands and narrow junction "none" lanes next to the
    # reference line (curb strips, checked against CARLA renders). Surfaces above others (bridges) form an upper layer. Holes are filled (junction gaps, slivers) unless
    # they are large off-road areas, contain sidewalks or raised lanes, or are curbed islands.
    surface_keys = [
        k
        for k, ln in lanes.items()
        if ln["type"] != "sidewalk"
        and k not in raised
        and not (ln["type"] == "none" and ln["junction"] != -1 and abs(ln["id"]) == 1 and np.max(np.linalg.norm(ln["outer"][:, :2] - ln["inner"][:, :2], axis=1)) <= 1.0)
        and len(ln["inner"]) > 1
    ]
    outline_of = lambda ln: shapely.Polygon(np.vstack([ln["inner"], ln["outer"][::-1]]))
    surfaces = [outline_of(lanes[k]) for k in surface_keys]
    surfaces = [p if p.is_valid else max(shapely.get_parts(shapely.make_valid(p)), key=lambda q: q.area) for p in surfaces]
    blocking = shapely.STRtree(
        [shapely.force_2d(outline_of(ln)) for k, ln in lanes.items() if (ln["type"] == "sidewalk" or (k in raised and k not in curb_lanes)) and len(ln["inner"]) > 1]
    )
    blocking_area = shapely.union_all(shapely.make_valid(blocking.geometries))
    curb_zone = shapely.union_all(shapely.buffer(curbs, 0.2))
    junction_zone = shapely.union_all(
        shapely.buffer([shapely.force_2d(p) for k, p in zip(surface_keys, surfaces) if lanes[k]["junction"] != -1], 0.2)
    )

    def keep(hole):
        radius = shapely.maximum_inscribed_circle(hole, 0.1).length
        curbed = hole.exterior.intersection(curb_zone).length >= ISLAND_CURB_FRACTION * hole.exterior.length
        blocked = len(blocking.query(hole.buffer(-0.1), predicate="intersects")) > 0
        # NOTE: Holes enclosed by regular roads (not junction gaps) are islands, e.g. roundabout centres.
        in_junction = hole.exterior.intersection(junction_zone).length >= 0.5 * hole.exterior.length
        return radius >= KEEP_HOLE_RADIUS or blocked or (radius >= ISLAND_MIN_RADIUS and (curbed or not in_junction))

    z_near = lambda polygon, xy: shapely.get_coordinates(polygon.exterior, include_z=True)[np.argmin(np.linalg.norm(shapely.get_coordinates(polygon.exterior) - xy, axis=1)), 2]
    upper = set()
    for i, j in zip(*shapely.STRtree(surfaces).query(surfaces, predicate="overlaps")):
        overlap = shapely.intersection(surfaces[i], surfaces[j])
        if i < j and overlap.area > 0.5:
            c = np.array(overlap.centroid.coords[0][:2])
            dz = z_near(surfaces[i], c) - z_near(surfaces[j], c)
            if abs(dz) > LAYER_DZ:
                upper.add(i if dz > 0 else j)

    def lifter(polygons):
        # Heights come from the nearest lane vertex of the same layer.
        points = np.vstack([shapely.get_coordinates(p.exterior, include_z=True) for p in polygons] or [np.zeros((0, 3))])
        kd = cKDTree(points[:, :2]) if len(points) else None
        return lambda xy: np.column_stack([xy, points[kd.query(xy)[1], 2]])

    def closed_union(polygons):
        union = shapely.union_all(shapely.force_2d(polygons), grid_size=0.001)
        return union.buffer(CLOSING, join_style="mitre").buffer(-CLOSING, join_style="mitre")

    driving_tree = shapely.STRtree(
        [shapely.force_2d(outline_of(ln)).buffer(-0.05) for ln in lanes.values() if ln["type"] == "driving" and len(ln["inner"]) > 1]
    )
    regions, lifts, fillers, dropped = [], [], [], []
    for layer in ([s for n, s in enumerate(surfaces) if n not in upper], [surfaces[n] for n in upper]):
        lift = lifter(layer)
        parts = shapely.get_parts(closed_union(layer))
        # NOTE: Tiny pieces and thin pieces without any driving lane (shoulder rings around islands) are not road.
        thin = lambda p: not driving_tree.query(p, predicate="intersects").size and shapely.maximum_inscribed_circle(p, 0.05).length < 1.0
        small = [p for p in parts if p.area < MIN_SURFACE_AREA or thin(p)]
        dropped += small
        union = shapely.union_all([p for p in parts if not any(p is q for q in small)])
        # NOTE: Gaps narrower than 2 * GAP_CLOSING between road surfaces (junction wedges) are road, never over
        # sidewalks or raised lanes.
        gaps = union.buffer(GAP_CLOSING, join_style="mitre").buffer(-GAP_CLOSING, join_style="mitre")
        union = shapely.union_all([union, gaps.difference(blocking_area)])
        filled = [shapely.Polygon(r) for p in shapely.get_parts(union) for r in p.interiors if not keep(shapely.Polygon(r))]
        fillers += [shapely.Polygon(lift(shapely.get_coordinates(f.exterior))) for f in filled]
        regions.append(shapely.union_all([union] + filled))
        lifts.append(lift)

    # 4. Other lane types become outline-only surfaces (not raised and not part of a dropped surface piece). Sidewalks
    # and curb strips (the kerb between road edge and sidewalk) are walkways; adjacent ones of a section side merge.
    dropped = shapely.union_all(dropped)
    is_walkway = lambda k: (lanes[k]["type"] == "sidewalk" or k in curb_lanes) and len(lanes[k]["inner"]) > 1
    walkway_runs = []
    for key in sorted(filter(is_walkway, lanes), key=lambda k: (lanes[k]["road"], lanes[k]["section"], np.sign(lanes[k]["id"]), abs(lanes[k]["id"]))):
        prev = lanes[walkway_runs[-1][-1]] if walkway_runs else None
        ln = lanes[key]
        if prev and (prev["road"], prev["section"], prev["id"] + np.sign(ln["id"])) == (ln["road"], ln["section"], ln["id"]):
            walkway_runs[-1].append(key)
        else:
            walkway_runs.append([key])
    road_area = shapely.union_all(regions)
    for run in walkway_runs:
        first, last = lanes[run[0]], lanes[run[-1]]
        outline = np.vstack([first["inner"], last["outer"][::-1], first["inner"][:1]])
        polygon = shapely.make_valid(shapely.Polygon(outline[:, :2]))
        # NOTE: Junction sidewalk lanes crossing the carriageway are pedestrian crossings, not walkways.
        if polygon.intersection(road_area).area > 0.5 * polygon.area:
            continue
        yield Walkway(object_id=run[0], outline=Polyline3D.from_array(outline))
    for key, ln in lanes.items():
        surface = ln["type"] in SURFACE_TYPES and key not in raised and not polygon_of(ln).representative_point().within(dropped)
        if surface and not is_walkway(key) and len(ln["inner"]) > 1:
            outline = np.vstack([ln["inner"], ln["outer"][::-1], ln["inner"][:1]])
            yield SURFACE_TYPES[ln["type"]](object_id=key, outline=Polyline3D.from_array(outline))

    # 9. Road edges: boundaries of the filled surface of each layer, without the seams where layers join.
    runs = []
    for n, region in enumerate(regions):
        other = regions[1 - n]
        for ring in [r for p in shapely.get_parts(region) for r in [p.exterior, *p.interiors]]:
            coords = lifts[n](shapely.get_coordinates(ring))
            mid = 0.5 * (coords[:-1] + coords[1:])
            seam = np.zeros(len(mid), dtype=bool)
            if not other.is_empty:
                near = shapely.distance(other, shapely.points(mid[:, :2])) < 0.01
                seam = near & (np.abs(lifts[1 - n](mid[:, :2])[:, 2] - mid[:, 2]) < LAYER_DZ)
            cuts = np.flatnonzero(np.diff(np.concatenate([[1], seam, [1]])))
            runs += [coords[a : b + 1] for a, b in zip(cuts[::2], cuts[1::2]) if b > a]
    edges = shapely.line_merge(shapely.MultiLineString(runs))
    edges = [e for e in shapely.get_parts(edges) if e.length > 0.05]
    for n, edge in enumerate(split_line_geometry_by_max_length(edges, MAX_ROAD_EDGE_LENGTH)):
        yield RoadEdge(object_id=n, road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY, polyline=Polyline3D.from_linestring(edge))

    # 10. Remaining road-level surfaces (flush "none"/median lanes, filled holes) complete the drivable area.
    for key in surface_keys:
        if lanes[key]["type"] not in LANE_TYPES and lanes[key]["type"] not in SURFACE_TYPES:
            outline = np.vstack([lanes[key]["inner"], lanes[key]["outer"][::-1], lanes[key]["inner"][:1]])
            yield GenericDrivable(object_id=key, outline=Polyline3D.from_array(outline))
    for n, filler in enumerate(fillers):
        yield GenericDrivable(object_id=f"filler_{n}", outline=Polyline3D.from_array(shapely.get_coordinates(filler.exterior, include_z=True)))

    # 11. Intersections: outline of the union of all road-level lanes of the junction's connecting roads.
    by_junction = {}
    for key, polygon in zip(surface_keys, surfaces):
        if lanes[key]["junction"] != -1:
            by_junction.setdefault(lanes[key]["junction"], []).append(polygon)
    for junction_id, polygons in by_junction.items():
        group_ids = sorted({g for g in groups if roads[int(g.split("_")[0])]["junction"] == junction_id})
        if not group_ids:
            continue
        polygon = max(shapely.get_parts(closed_union(polygons)), key=lambda p: p.area)
        yield Intersection(
            object_id=junction_id,
            intersection_type=IntersectionType.DEFAULT,
            lane_group_ids=group_ids,
            outline=Polyline3D.from_array(lifter(polygons)(shapely.get_coordinates(polygon.exterior))),
        )
