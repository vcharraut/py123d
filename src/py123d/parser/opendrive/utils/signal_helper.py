from typing import Any, Dict, Final, List, Tuple

from py123d.datatypes import StopZoneType
from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.xodr_parser.lane import section_at

_STOP_ZONE_TYPES: Final[Dict[str, StopZoneType]] = {
    "1000001": StopZoneType.TRAFFIC_LIGHT,
    "206": StopZoneType.STOP_SIGN,
    "205": StopZoneType.YIELD_SIGN,
}


def collect_stop_lines(
    roads: Dict[int, Dict[str, Any]],
    lanes: Dict[str, Dict[str, Any]],
    all_successors: Dict[str, List[str]],
    all_predecessors: Dict[str, List[str]],
) -> Dict[Tuple[str, bool], StopZoneType]:
    """Collects stop lines across the lane where it enters the junction (CARLA BasicAgent stop waypoint), found from
    the lanes each traffic light / stop / yield signal is valid for.

    :param roads: Parsed roads, keyed by id.
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param all_successors: Successors over all lanes.
    :param all_predecessors: Predecessors over all lanes.
    :return: Stop zone type per (lane id, stop line at the lane start instead of its end).
    """
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
    return stop_lines
