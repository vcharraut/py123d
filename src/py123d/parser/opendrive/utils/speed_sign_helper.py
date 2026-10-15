import heapq
import json
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np

from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.xodr_parser.lane import section_at

# NOTE: Exported from CARLA 0.9.16: (road id, lane id, s, limit in km/h) of every lane on which a vehicle takes the
# limit of a traffic.speed_limit actor, at the first point in driving direction where a car put there takes it (one
# position per sign, road and driving direction).
_SPEED_SIGNS_FILE: Final[Path] = Path(__file__).parents[1] / "carla_maps" / "speed_signs.json"
_DEFAULT_SPEED_LIMIT: Final[float] = 30.0 / 3.6  # [m/s], CARLA limit of a vehicle that has not passed a speed sign
_MIN_SECTION_LENGTH: Final[float] = 5.0  # [m], signs closer than this to a lane section border act from that border


def load_speed_signs(xodr_file: Path) -> Optional[List[Tuple[int, int, float, float]]]:
    """Loads the speed signs CARLA places on a town map.

    :param xodr_file: Path to the OpenDRIVE (.xodr or .xodr.gz) file.
    :return: (road id, lane id, s, limit in m/s) per lane a sign acts on, None for maps without exported signs.
    """
    signs = json.loads(_SPEED_SIGNS_FILE.read_text()).get(xodr_file.name.split(".")[0])
    return None if signs is None else [(int(r), int(i), float(s), float(kmh) / 3.6) for r, i, s, kmh in signs]


def _sign_border(road: Dict[str, Any], s: float) -> float:
    # The position a sign acts from: the nearest lane section border (or road end) if close enough, else s itself.
    borders = np.array([section["s"] for section in road["sections"]] + [road["length"]])
    nearest = float(borders[np.argmin(np.abs(borders - s))])
    return nearest if abs(nearest - s) < _MIN_SECTION_LENGTH else s


def split_sections_at_speed_signs(
    roads: Dict[int, Dict[str, Any]], signs: Optional[List[Tuple[int, int, float, float]]]
) -> None:
    """Splits lane sections at the speed signs, so that a lane carries one sign limit over its whole length.

    Junction roads are not split: signals refer to their lanes as a whole (see collect_stop_lines).

    :param roads: Parsed roads, keyed by id (modified in place).
    :param signs: Speed signs (see load_speed_signs).
    """
    for road_id, s in sorted({(road_id, s) for road_id, _, s, _ in signs or []}):
        road, sections = roads[road_id], roads[road_id]["sections"]
        border = _sign_border(road, s)
        if road["junction"] != -1 or border == road["length"] or any(sec["s"] == border for sec in sections):
            continue
        k = section_at(road, border)
        lanes = sections[k]["lanes"]
        # NOTE: Width and mark records hold absolute s, so both halves keep them; marks of the second half start at
        # the split (road lines are drawn from the mark's s on).
        sections[k] = {"s": sections[k]["s"], "lanes": {i: {**lane, "succ": i} for i, lane in lanes.items()}}
        second = {
            i: {
                **lane,
                "pred": i,
                "marks": [
                    {**mark, "s": max(mark["s"], border)}
                    for j, mark in enumerate(lane["marks"])
                    if j + 1 == len(lane["marks"]) or lane["marks"][j + 1]["s"] > border
                ],
            }
            for i, lane in lanes.items()
        }
        sections.insert(k + 1, {"s": border, "lanes": second})


def collect_sign_speed_limits(
    roads: Dict[int, Dict[str, Any]],
    lanes: Dict[str, Dict[str, Any]],
    is_lane: Dict[str, None],
    all_successors: Dict[str, List[str]],
    successors: Dict[str, List[str]],
    signs: Optional[List[Tuple[int, int, float, float]]],
) -> Dict[str, Optional[float]]:
    """Computes the speed limit a CARLA vehicle carries on each lane: the limit of the last speed sign passed.

    :param roads: Parsed roads, keyed by id (split at the signs, see split_sections_at_speed_signs).
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param is_lane: Ordered ids of py123d lanes.
    :param all_successors: Successors over all lanes.
    :param successors: Successors of each py123d lane.
    :param signs: Speed signs (see load_speed_signs).
    :return: Sign speed limit in m/s per py123d lane, None without exported signs.
    """
    if signs is None:
        return dict.fromkeys(is_lane)

    # NOTE: A sign holds from its position on in driving direction (lanes with id < 0 follow the road direction): on
    # the lane starting there, or on the next lanes if the road ends there or is a junction road (not split).
    starts = {}
    for road_id, lane_id, s, limit in signs:
        road = roads[road_id]
        border = _sign_border(road, s)
        if lane_id < 0:
            on_road = border < road["length"]
            key = lane_key(road_id, section_at(road, border) if on_road else len(road["sections"]) - 1, lane_id)
        else:
            on_road = border > 0.0
            key = lane_key(road_id, section_at(road, border - 1e-6) if on_road else 0, lane_id)
        keys = [key] if on_road and road["junction"] == -1 else all_successors.get(key, [])
        starts.update({key: limit for key in keys if key in is_lane})

    # NOTE: The limit travels downstream until the next sign. In CARLA it depends on the way the vehicle came; where
    # several signs reach a lane, the nearest one wins. A vehicle that has passed no sign has CARLA's default limit.
    length = lambda key: float(
        np.linalg.norm(np.diff(0.5 * (lanes[key]["inner"] + lanes[key]["outer"])[:, :2], axis=0), axis=1).sum()
    )
    limits = dict.fromkeys(is_lane, _DEFAULT_SPEED_LIMIT)
    queue = [(0.0, key, limit) for key, limit in starts.items()]
    heapq.heapify(queue)
    done = set()
    while queue:
        distance, key, limit = heapq.heappop(queue)
        if key in done:
            continue
        done.add(key)
        limits[key] = limit
        for successor in successors[key]:
            if successor not in done and successor not in starts:
                heapq.heappush(queue, (distance + length(key), successor, limit))
    return limits
