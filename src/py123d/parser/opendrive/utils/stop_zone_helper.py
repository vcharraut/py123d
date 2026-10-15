import json
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.datatypes import BaseMapObject, StopZone, StopZoneType
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.xodr_parser.lane import eval_borders, section_at

_STOP_LINE_DEPTH: Final[float] = 0.5  # [m]
_STOP_LINE_SEARCH: Final[float] = 15.0  # [m], upstream distance searched for painted stop lines / crosswalks
_STOP_LINE_JUNCTION_SEARCH: Final[float] = 20.0  # [m], painted stop lines inside junctions lie close to the entry
_STOP_LINE_MAX_TURN: Final[float] = 30.0  # [deg], ... before the junction lane turns
_STOP_LINE_MIN_ANGLE: Final[float] = 60.0  # [deg], painted stop lines must cross the lane at least this steeply
_STOP_LINE_MERGE: Final[float] = 0.2  # [m], stop lines of one signal group or sign closer than this are one line
# NOTE: CARLA counts a stop at a stop sign only while the vehicle's centre is in the sign's affected stretch; a vehicle
# stopping at the line has its centre this far behind it (CARLA leaderboard hero vehicle, Lincoln MKZ 2020).
_VEHICLE_HALF_LENGTH: Final[float] = 2.45  # [m]
_REACH_MARGIN: Final[float] = 0.5  # [m], kept inside the affected stretch
# NOTE: CARLA's red-light test fires when the vehicle's tail (from 0.8 half lengths behind its centre) crosses the line
# at the junction entry: a painted line further past the entry than this is no traffic-light stop line.
_RED_LIGHT_PAST: Final[float] = 1.8 * _VEHICLE_HALF_LENGTH - _REACH_MARGIN  # [m]
_YIELD_BACK: Final[float] = 1.0  # [m], yield lines lie this far before the first lane of another road they meet ...
_YIELD_MERGE: Final[float] = 10.0  # [m], ... where it is at least this far into the junction (merge ramps)
# NOTE: Exported from CARLA 0.9.16 (scripts/opendrive_parity/export_stop_signs.py), in the OpenDRIVE frame: ("stop" |
# "yield", x, y, z, [(road id, lane id, s, first, last, road id, lane id, s)]) of every traffic.stop / traffic.yield
# actor: its trigger volume centre and, per approach lane, the lane's last position before the junction, the distances
# before it of the first and last lane centre positions where scenario_runner's RunningStopTest counts a vehicle as
# affected (negative: inside the junction), and the last of these positions.
_CARLA_SIGNS_FILE: Final[Path] = Path(__file__).parents[1] / "carla_maps" / "stop_signs.json"
_CARLA_SIGN_TYPES: Final[Dict[str, StopZoneType]] = {"stop": StopZoneType.STOP_SIGN, "yield": StopZoneType.YIELD_SIGN}


def iter_stop_zones(
    roads: Dict[int, Dict[str, Any]],
    lanes: Dict[str, Dict[str, Any]],
    is_lane: Dict[str, None],
    all_successors: Dict[str, List[str]],
    stop_lines: Dict[Tuple[str, bool], Tuple[StopZoneType, Optional[str], Optional[int]]],
    crosswalk_tree: shapely.STRtree,
    painted_stop_lines: List[shapely.Polygon],
    painted_tree: shapely.STRtree,
    painted_directions: npt.NDArray[np.float64],
    carla_signs: List[List[Any]],
) -> Iterator[BaseMapObject]:
    """Yields stop zones of signal stop lines, CARLA stop / yield signs and painted stop lines without a signal.

    The stop lines of one traffic-light signal group or one CARLA sign that lie side by side are merged into one stop
    zone.

    :param roads: Parsed roads, keyed by id.
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param is_lane: Ordered ids of py123d lanes.
    :param all_successors: Successors over all lanes.
    :param stop_lines: Stop zone type, signal group id and signal sequence per (lane id, stop line at the lane start)
        (see collect_stop_lines).
    :param crosswalk_tree: Tree of the crosswalk polygons.
    :param painted_stop_lines: Painted stop line areas.
    :param painted_tree: Tree of the painted stop line areas.
    :param painted_directions: (n, 2) unit directions of the painted stop lines.
    :param carla_signs: CARLA stop and yield signs (see load_stop_signs), replacing the OpenDRIVE ones.
    """

    def section_range(lane: Dict[str, Any]) -> Tuple[float, float]:
        sections, road = roads[lane["road"]]["sections"], roads[lane["road"]]
        s_end = sections[lane["section"] + 1]["s"] if lane["section"] + 1 < len(sections) else road["length"]
        return sections[lane["section"]]["s"], s_end

    def lane_hits(
        key: str,
        from_end: bool,
        tree: shapely.STRtree,
        length: float = _STOP_LINE_SEARCH,
        directions: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        # s positions (0.1 m apart, walking away from the lane end or start) whose lane centre lies in a tree geometry.
        # With line directions, only lines crossing the lane at more than _STOP_LINE_MIN_ANGLE count.
        lane, (s0, s1) = lanes[key], section_range(lanes[key])
        forward = lane["id"] <= 0
        s_from = (s1 if forward else s0) if from_end else (s0 if forward else s1)
        s_near = np.linspace(s_from, s_from + (-1 if forward == from_end else 1) * length, int(length / 0.1) + 1)
        s_near = s_near[(s_near >= s0) & (s_near <= s1)]
        borders = eval_borders(roads[lane["road"]], lane["section"], s_near)
        if len(s_near) < 2:
            return np.zeros(0)
        centres = 0.5 * (borders[lane["id"]] + borders[lane["id"] - np.sign(lane["id"])])[:, :2]
        points, geometries = tree.query(shapely.points(centres), predicate="intersects")
        if directions is not None:
            tangents = np.gradient(centres, axis=0)
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
            alignment = np.abs(np.sum(tangents[points] * directions[geometries], axis=1))
            crossing = alignment < np.cos(np.radians(_STOP_LINE_MIN_ANGLE))
            # NOTE: The lane must not have turned yet (painted lines of other approaches cross turning lanes).
            straight = np.sum(tangents[points] * tangents[0], axis=1) > np.cos(np.radians(_STOP_LINE_MAX_TURN))
            points = points[crossing & straight]
        return s_near[np.unique(points)]

    def stop_zone(
        object_id: str,
        zone_type: StopZoneType,
        key: str,
        s_stop: float,
        controlled: List[str],
        painted: bool = False,
        signal_group_id: Optional[str] = None,
        signal_sequence: Optional[int] = None,
    ) -> StopZone:
        # NOTE: The 0.5 m strip lies upstream of the stop position, across the full lane width (oriented along the
        # painted line if there is one). The controlled lanes start at the stop line (junction lanes), like
        # traffic-light lane connectors.
        lane, (s0, s1) = lanes[key], section_range(lanes[key])
        upstream = -np.sign(0.5 - lane["id"]) * _STOP_LINE_DEPTH
        s_pair = np.clip(np.array([s_stop + upstream, s_stop]), s0, s1)
        borders = eval_borders(roads[lane["road"]], lane["section"], s_pair)
        inner, outer = borders[lane["id"] - np.sign(lane["id"])], borders[lane["id"]]
        if painted:
            centre = 0.5 * (inner[1] + outer[1])
            across = painted_directions[painted_tree.nearest(shapely.Point(centre[:2]))]
            across = across * np.sign(np.dot(across, outer[1, :2] - inner[1, :2]))
            back = np.array([-across[1], across[0]])
            back = back * (1.0 if np.dot(back, inner[0, :2] - inner[1, :2]) >= 0 else -1.0)
            half = 0.5 * np.linalg.norm(outer[1, :2] - inner[1, :2])
            ends = np.array([centre[:2] - half * across, centre[:2] + half * across])
            inner = np.column_stack([np.vstack([ends[0] + _STOP_LINE_DEPTH * back, ends[0]]), inner[:, 2]])
            outer = np.column_stack([np.vstack([ends[1] + _STOP_LINE_DEPTH * back, ends[1]]), outer[:, 2]])
        return StopZone(
            object_id=object_id,
            stop_zone_type=zone_type,
            outline=Polyline3D.from_array(np.vstack([inner, outer[::-1], inner[:1]])),
            lane_ids=controlled,
            # NOTE: The controlled lanes are junction lanes, whose junction is the stop zone's intersection.
            intersection_id=lanes[controlled[0]]["junction"] if lanes[controlled[0]]["junction"] != -1 else None,
            signal_group_id=signal_group_id,
            signal_sequence=signal_sequence,
        )

    # NOTE: (lane id, stop line at the lane start, type, signal group, sequence, merge group, CARLA reach: range of
    # distances of the line before the junction entry and the lane id and s of its last position, or None). CARLA signs
    # act on the approach lanes that reach them.
    traffic_light = StopZoneType.TRAFFIC_LIGHT
    entries = [
        (key, at_start, zone_type, group, sequence, ("signal", group) if zone_type == traffic_light else None, None)
        for (key, at_start), (zone_type, group, sequence) in stop_lines.items()
    ]
    for n, (kind, _, _, _, approaches) in enumerate(carla_signs):
        for road_id, lane_id, s, first, last, end_road, end_lane, end_s in approaches:
            key = lane_key(road_id, section_at(roads[road_id], s), lane_id)
            end_key = lane_key(end_road, section_at(roads[end_road], end_s), end_lane)
            # NOTE: CARLA scores stop signs only: the vehicle's centre must stop in the affected stretch.
            margin = min(_REACH_MARGIN, (first - last) / 2)
            lo, hi = last + margin - _VEHICLE_HALF_LENGTH, first - margin - _VEHICLE_HALF_LENGTH
            reach = (lo, hi, last, first, end_key, end_s) if kind == "stop" and end_key in lanes else None
            if key in is_lane:
                entries.append((key, False, _CARLA_SIGN_TYPES[kind], None, None, ("sign", n), reach))

    predecessors = {}
    for key, nexts in all_successors.items():
        for nxt in nexts:
            predecessors.setdefault(nxt, []).append(key)
    junction_polygons = {
        k: shapely.Polygon(np.vstack([lane["inner"], lane["outer"][::-1]])[:, :2]).buffer(0)
        for k, lane in lanes.items()
        if lane["junction"] != -1 and lane["type"] == "driving" and len(lane["inner"]) > 1
    }

    def conflict_point(approach: str, key: str) -> Optional[Tuple[float, str, float]]:
        # Where the path through the junction starting with a junction lane first overlaps a junction lane not reachable
        # from the approach: (distance along the path, lane id, s on it).
        junction = lanes[key]["junction"]
        own, todo = set(), [approach]
        while todo:
            for nxt in all_successors.get(todo.pop(), []):
                if nxt not in own and lanes[nxt]["junction"] == junction:
                    own.add(nxt)
                    todo.append(nxt)
        others = [p for k, p in junction_polygons.items() if lanes[k]["junction"] == junction and k not in own]
        path, walked = [key], 0.0
        while len(path) < 20 and all_successors.get(path[-1]):
            nxt = all_successors[path[-1]][0]
            if lanes[nxt]["junction"] != junction or nxt in path:
                break
            path.append(nxt)
        if not others:
            return None
        other_area = shapely.union_all(others)
        for k in path:
            lane = lanes[k]
            forward = lane["id"] <= 0
            s0, s1 = section_range(lane)
            # NOTE: Lane borders run in driving direction.
            centre = 0.5 * (lane["inner"] + lane["outer"])[:, :2]
            half = 0.5 * np.linalg.norm(lane["outer"][:, :2] - lane["inner"][:, :2], axis=1)
            steps = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
            hit = shapely.distance(other_area, shapely.points(centre)) < half - 0.1
            if hit.any():
                i = int(np.argmax(hit))
                along = steps[i] / max(steps[-1], 1e-9) * (s1 - s0)
                return walked + steps[i], k, s0 + along if forward else s1 - along
            walked += steps[-1]
        return None

    signal_zones, merge_groups = [], {}
    for key, at_start, zone_type, signal_group_id, signal_sequence, merge_group, reach in entries:
        controlled = [key] if at_start else [s for s in all_successors[key] if s in is_lane] or [key]
        # NOTE: The vehicle stops at the painted stop line (on the approach or the first junction lane), else at the
        # junction entry (CARLA BasicAgent stop waypoint, and CARLA's red-light line). Only OpenDRIVE stop and yield
        # signs fall back to a crosswalk on the approach. Painted lines past the entry do not count for a CARLA stop
        # sign that must be stopped at before it.
        painted_before = (
            lane_hits(key, True, painted_tree, directions=painted_directions) if not at_start else np.zeros(0)
        )
        past = _RED_LIGHT_PAST if zone_type == traffic_light else _STOP_LINE_JUNCTION_SEARCH
        painted_after = (
            lane_hits(controlled[0], False, painted_tree, past, painted_directions)
            if (controlled[0] != key or at_start) and (reach is None or reach[0] < 0)
            else np.zeros(0)
        )
        crossing = (
            lane_hits(key, True, crosswalk_tree)
            if not at_start and zone_type != traffic_light and merge_group is None
            else np.zeros(0)
        )
        painted = len(painted_before) + len(painted_after) > 0
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
            forward = lanes[stop_key]["id"] <= 0
            s_stop = (s1 if forward else s0) if not at_start else (s0 if forward else s1)
        if zone_type == StopZoneType.YIELD_SIGN and not painted and not at_start:
            # NOTE: Without a painted line, a vehicle yields at the junction entry, or on merge ramps (whose junction
            # starts long before the merge) where its lane first meets a lane coming from another road.
            conflicts = [point for c in controlled if (point := conflict_point(key, c)) is not None]
            if conflicts and min(conflicts)[0] >= _YIELD_MERGE:
                _, stop_key, s_hit = min(conflicts)
                s_stop = s_hit - _YIELD_BACK * (1.0 if lanes[stop_key]["id"] <= 0 else -1.0)
        if reach is not None:
            # NOTE: The line moves into CARLA's reach, given in distances before the junction entry (negative: past it,
            # on the first junction lane).
            sense = lambda k: 1.0 if lanes[k]["id"] <= 0 else -1.0
            lane_end = lambda k: section_range(lanes[k])[1 if lanes[k]["id"] <= 0 else 0]
            lane_start = lambda k: section_range(lanes[k])[0 if lanes[k]["id"] <= 0 else 1]
            lo, hi, last, first, end_key, end_s = reach
            if stop_key == key:
                before = (lane_end(key) - s_stop) * sense(key)
            else:
                before = (lane_start(stop_key) - s_stop) * sense(stop_key)
            clamped = min(max(before, lo), hi)
            # NOTE: A line just past the entry would leave no room for the strip on the junction lane: it stays at the
            # entry if CARLA's stretch (without the margin) still holds the vehicle there.
            if -_STOP_LINE_DEPTH < clamped < 0 and last <= _VEHICLE_HALF_LENGTH <= first:
                clamped = 0.0
            if clamped != before:
                painted = False
                # NOTE: Past the entry, the line goes on the junction lane of the affected stretch, from its last
                # position.
                if clamped >= 0:
                    # NOTE: Short approach lanes hand the rest of the distance on to the lane before them.
                    stop_key, rest = key, clamped
                    while abs(lane_end(stop_key) - lane_start(stop_key)) < rest and predecessors.get(stop_key):
                        rest -= abs(lane_end(stop_key) - lane_start(stop_key))
                        stop_key = predecessors[stop_key][0]
                    s_stop = lane_end(stop_key) - rest * sense(stop_key)
                else:
                    stop_key, s_stop = end_key, end_s + (last - clamped) * sense(end_key)
        sign = merge_group is not None and merge_group[0] == "sign"
        object_id = f"carla_{zone_type.name.lower()}_{merge_group[1]}_{key}" if sign else f"stop_{key}"
        zone = stop_zone(
            object_id + ("_start" if at_start else ""),
            zone_type,
            stop_key,
            s_stop,
            controlled,
            painted,
            signal_group_id,
            signal_sequence,
        )
        signal_zones.append(zone.shapely_polygon)
        if merge_group is not None:
            merge_groups.setdefault(merge_group, []).append(zone)
            continue
        yield zone

    # NOTE: The stop lines of one signal group or sign that lie side by side are one stop line across the road: one stop
    # zone controlling the lanes of all of them. Closing (buffer out, then in) bridges the gaps between their strips.
    close = lambda polygon: polygon.buffer(_STOP_LINE_MERGE, join_style="mitre").buffer(
        -_STOP_LINE_MERGE, join_style="mitre"
    )
    closed = {
        group: close(shapely.union_all([zone.shapely_polygon for zone in zones]))
        for group, zones in merge_groups.items()
    }
    merged_lines = [
        (merge_groups[group], line)
        for group, area in closed.items()
        for line in shapely.get_parts(area)
        if not line.is_empty
    ]
    for zones, line in merged_lines:
        members = [zone for zone in zones if line.intersects(zone.shapely_polygon.representative_point())]
        if len(members) == 1:
            yield members[0]
            continue
        corners = np.vstack([zone.outline.array for zone in members])
        xy = np.array(line.simplify(0.01).exterior.coords)
        heights = corners[np.argmin(np.linalg.norm(xy[:, None] - corners[None, :, :2], axis=2), axis=1), 2]
        yield StopZone(
            object_id=members[0].object_id,
            stop_zone_type=members[0].stop_zone_type,
            outline=Polyline3D.from_array(np.column_stack([xy, heights])),
            lane_ids=list(dict.fromkeys(lane_id for zone in members for lane_id in zone.lane_ids)),
            intersection_id=members[0].intersection_id,
            signal_group_id=members[0].signal_group_id,
            signal_sequence=members[0].signal_sequence,
        )
    # NOTE: Zero-depth strips (stop position clipped at the lane end) vanish in the closing and stay as they are.
    yield from (
        zone
        for group, zones in merge_groups.items()
        for zone in zones
        if not closed[group].intersects(zone.shapely_polygon.representative_point())
    )

    # NOTE: Painted stop lines without a signal are stop lines as well.
    signal_tree = shapely.STRtree(signal_zones)
    driving_keys = [k for k in is_lane if lanes[k]["type"] == "driving"]
    centre_tree = shapely.STRtree(
        [shapely.LineString(0.5 * (lanes[k]["inner"] + lanes[k]["outer"])[:, :2]) for k in driving_keys]
    )
    for n, painted in enumerate(painted_stop_lines):
        if len(signal_tree.query(painted, predicate="intersects")):
            continue
        for key in (driving_keys[j] for j in centre_tree.query(painted, predicate="intersects")):
            in_junction = lanes[key]["junction"] != -1
            # NOTE: Stop lines guard junction entries: approach lanes must lead into a junction.
            if not in_junction and all(lanes[s]["junction"] == -1 for s in all_successors[key]):
                continue
            length = _STOP_LINE_JUNCTION_SEARCH if in_junction else _STOP_LINE_SEARCH
            hits = lane_hits(key, not in_junction, shapely.STRtree([painted]), length, painted_directions[n : n + 1])
            controlled = [key] if in_junction else [s for s in all_successors[key] if s in is_lane] or [key]
            if len(hits):
                yield stop_zone(
                    f"painted_{n}_{key}",
                    StopZoneType.STOP_SIGN,
                    key,
                    np.median(hits),
                    controlled,
                    painted=True,
                )


def load_stop_signs(town: Optional[str]) -> Optional[List[List[Any]]]:
    """Loads the stop and yield signs CARLA places on a town map.

    :param town: Town name, i.e. the OpenDRIVE file name without suffixes.
    :return: Signs (see _CARLA_SIGNS_FILE), None for maps without exported signs.
    """
    return json.loads(_CARLA_SIGNS_FILE.read_text()).get(town)
