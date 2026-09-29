from typing import Any, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.datatypes import BaseMapObject, StopZone, StopZoneType
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.xodr_parser.lane import eval_borders

_STOP_LINE_DEPTH: Final[float] = 0.5  # [m]
_STOP_LINE_SEARCH: Final[float] = 15.0  # [m], upstream distance searched for painted stop lines / crosswalks
_STOP_LINE_JUNCTION_SEARCH: Final[float] = 20.0  # [m], painted stop lines inside junctions lie close to the entry
_STOP_LINE_MAX_TURN: Final[float] = 30.0  # [deg], ... before the junction lane turns
_STOP_LINE_MIN_ANGLE: Final[float] = 60.0  # [deg], painted stop lines must cross the lane at least this steeply


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
) -> Iterator[BaseMapObject]:
    """Yields stop zones of signal stop lines and of painted stop lines without a signal.

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

    signal_zones = []
    for (key, at_start), (zone_type, signal_group_id, signal_sequence) in stop_lines.items():
        controlled = [key] if at_start else [s for s in all_successors[key] if s in is_lane] or [key]
        # NOTE: The vehicle stops at the painted stop line (on the approach or the first junction lane), else before a
        # crosswalk on the approach, else at the junction entry (CARLA BasicAgent stop waypoint).
        painted_before = (
            lane_hits(key, True, painted_tree, directions=painted_directions) if not at_start else np.zeros(0)
        )
        painted_after = (
            lane_hits(controlled[0], False, painted_tree, _STOP_LINE_JUNCTION_SEARCH, painted_directions)
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
            forward = lanes[stop_key]["id"] <= 0
            s_stop = (s1 if forward else s0) if not at_start else (s0 if forward else s1)
        zone = stop_zone(
            f"stop_{key}{'_start' if at_start else ''}",
            zone_type,
            stop_key,
            s_stop,
            controlled,
            len(painted_before) + len(painted_after) > 0,
            signal_group_id,
            signal_sequence,
        )
        signal_zones.append(zone.shapely_polygon)
        yield zone

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
