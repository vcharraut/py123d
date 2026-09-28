from typing import Any, Dict, Final, Generator, List, Set, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.datatypes import BaseMapObject, RoadLine, RoadLineType
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.utils.id_system import lane_key
from py123d.parser.opendrive.xodr_parser.lane import eval_borders

_ROAD_LINE_TYPES: Final[Dict[Tuple[str, str], RoadLineType]] = {
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


def iter_road_lines(
    roads: Dict[int, Dict[str, Any]],
    sampled: Dict[Tuple[int, int], Tuple[npt.NDArray[np.float64], Dict[int, npt.NDArray[np.float64]]]],
    raised: Set[str],
) -> Generator[BaseMapObject, None, List[shapely.LineString]]:
    """Yields road lines from road marks, each mark lies on the outer border of its lane (lane 0: reference line).

    :param roads: Parsed roads, keyed by id.
    :param sampled: s-grid and lane borders per (road id, section index) (see collect_lanes).
    :param raised: Ids of raised lanes.
    :return: 2D curb lines, returned once all road lines are yielded.
    """
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
                    road_line_type=_ROAD_LINE_TYPES[mark["type"], color],
                    polyline=Polyline3D.from_array(points),
                )
    return curbs
