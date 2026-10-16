import xml.etree.ElementTree as ET
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np

from py123d.datatypes import StopZone, StopZoneType, TrafficLight
from py123d.geometry import EulerAngles, PoseSE3, Quaternion
from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.xodr_parser.lane import section_at
from py123d.parser.opendrive.xodr_parser.reference import eval_poly, eval_reference

_STOP_ZONE_TYPES: Final[Dict[str, StopZoneType]] = {
    "1000001": StopZoneType.TRAFFIC_LIGHT,
    "206": StopZoneType.STOP_SIGN,
    "205": StopZoneType.YIELD_SIGN,
}


def collect_stop_lines(
    roads: Dict[int, Dict[str, Any]],
    junctions: Dict[int, ET.Element],
    controllers: Dict[int, List[str]],
    lanes: Dict[str, Dict[str, Any]],
    all_successors: Dict[str, List[str]],
    all_predecessors: Dict[str, List[str]],
) -> Dict[Tuple[str, bool], Tuple[StopZoneType, Optional[str], Optional[int], List[str]]]:
    """Collects stop lines across the lane where it enters the junction (CARLA BasicAgent stop waypoint), found from
    the lanes each traffic light / stop / yield signal is valid for.

    :param roads: Parsed roads, keyed by id.
    :param junctions: Raw junction elements, keyed by id.
    :param controllers: Signal ids of each controller, keyed by id.
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param all_successors: Successors over all lanes.
    :param all_predecessors: Predecessors over all lanes.
    :return: Stop zone type, signal group id and signal sequence of the first signal, and the ids of all signals, per
        (lane id, stop line at the lane start instead of its end).
    """
    # NOTE: A controller switches its signals together (signal group), the junction lists its controllers in cycle
    # order (sequence). CARLA spawns no traffic light for a traffic-light signal without a controller.
    controller_of = {}
    for controller_id, signal_ids in controllers.items():
        for signal_id in signal_ids:
            controller_of.setdefault(signal_id, controller_id)
    sequence_of = {
        int(c.get("id")): int(c.get("sequence"))
        for junction in junctions.values()
        for c in junction.findall("controller")
        if c.get("sequence") is not None
    }

    signal_types, references = {}, []
    for road_id, road in roads.items():
        for element in road["signals"] if road["signals"] is not None else []:
            if element.tag == "signal":
                signal_types[element.get("id")] = element.get("type")
            s = min(max(float(element.get("s")), 0.0), road["length"] - 1e-5)
            validities = [(int(v.get("fromLane")), int(v.get("toLane"))) for v in element.findall("validity")]
            if not validities:
                ids = road["sections"][section_at(road, s)]["lanes"].keys()
                default = (
                    ((1, max(ids)), element.get("orientation") != "-"),
                    ((min(ids), -1), element.get("orientation") != "+"),
                )
                validities = [v for v, use in default if use and v[0] <= v[1]]
            if not validities or any(v != (0, 0) for v in validities):
                references.append((element.get("id"), road_id, s, validities))

    stop_lines = {}
    for signal_id, road_id, s, validities in references:
        zone_type = _STOP_ZONE_TYPES.get(signal_types.get(signal_id))
        controller_id = controller_of.get(signal_id)
        if zone_type == StopZoneType.TRAFFIC_LIGHT and controller_id is None:
            continue
        group_id = f"controller_{controller_id}" if zone_type == StopZoneType.TRAFFIC_LIGHT else None
        sequence = sequence_of.get(controller_id) if group_id is not None else None
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
                stop_lines.setdefault((key, at_start), (zone_type, group_id, sequence, []))[3].append(signal_id)
    return stop_lines


def iter_traffic_lights(
    roads: Dict[int, Dict[str, Any]],
    controllers: Dict[int, List[str]],
    lanes: Dict[str, Dict[str, Any]],
    signal_zones: Dict[str, List[StopZone]],
) -> Iterator[TrafficLight]:
    """Yields one traffic light per controlled traffic-light signal, at the signal's s/t/zOffset on its road (for CARLA
    towns the signal post).

    :param roads: Parsed roads, keyed by id.
    :param controllers: Signal ids of each controller, keyed by id.
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param signal_zones: Stop zones of each traffic-light signal, keyed by signal id (see iter_stop_zones).
    """
    controlled = {signal_id for signal_ids in controllers.values() for signal_id in signal_ids}
    signals = [
        (road, element)
        for road in roads.values()
        for element in (road["signals"].findall("signal") if road["signals"] is not None else [])
        if _STOP_ZONE_TYPES.get(element.get("type")) == StopZoneType.TRAFFIC_LIGHT and element.get("id") in controlled
    ]
    for road, element in signals:
        signal_id = element.get("id")
        zones = signal_zones.get(signal_id, [])
        lane_ids = list(dict.fromkeys(lane_id for zone in zones for lane_id in zone.lane_ids))
        s = np.array([min(max(float(element.get("s")), 0.0), road["length"])])
        x, y, h = eval_reference(road, s)
        # NOTE: The housing faces the driving direction of the controlled lanes at their start (the stop line), else
        # the direction the signal is valid for. Lane borders run in driving direction.
        starts = [0.5 * (lanes[k]["inner"] + lanes[k]["outer"])[:2, :2] for k in lane_ids if len(lanes[k]["inner"]) > 1]
        direction = sum((b - a) / max(np.linalg.norm(b - a), 1e-9) for a, b in starts)
        yaw = np.arctan2(direction[1], direction[0]) if starts else h[0] + (element.get("orientation") == "-") * np.pi
        # NOTE: t is relative to the reference line (not the lane offset); the signal position is the housing bottom.
        height = float(element.get("height", 0.0)) or TrafficLight.DEFAULT_HEIGHT
        width = float(element.get("width", 0.0)) or None
        z = eval_poly(road["elevation"], s)[0] + float(element.get("zOffset", 0.0)) + height / 2
        t = float(element.get("t", 0.0))
        quaternion = Quaternion.from_euler_angles(EulerAngles(0.0, 0.0, float(yaw)))
        yield TrafficLight.from_pose(
            object_id=signal_id,
            pose_se3=PoseSE3(
                float(x[0] - t * np.sin(h[0])),
                float(y[0] + t * np.cos(h[0])),
                float(z),
                quaternion.qw,
                quaternion.qx,
                quaternion.qy,
                quaternion.qz,
            ),
            width=width,
            height=height,
            lane_ids=lane_ids,
            stop_zone_ids=[zone.object_id for zone in zones],
            intersection_id=zones[0].intersection_id if zones else None,
        )
