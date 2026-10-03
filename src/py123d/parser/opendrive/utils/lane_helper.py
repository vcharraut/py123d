from typing import Any, Dict, Final, Generator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from py123d.datatypes import BaseMapObject, Lane, LaneGroup, LaneType
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.utils.id_system import lane_key

LANE_TYPES: Final[Dict[str, LaneType]] = {"driving": LaneType.SURFACE_STREET, "biking": LaneType.BIKE_LANE}
MIN_LANE_WIDTH: Final[float] = 0.1  # [m], narrower lane samples do not exist yet (entry lanes) / anymore (exit lanes)


def iter_lanes_and_lane_groups(
    lanes: Dict[str, Dict[str, Any]],
    is_lane: Dict[str, None],
    predecessors: Dict[str, List[str]],
    successors: Dict[str, List[str]],
    openings: Dict[str, List[Tuple[str, bool]]],
) -> Generator[BaseMapObject, None, Dict[str, List[str]]]:
    """Yields lanes and lane groups: one group per road, section and side, lanes ordered inner to outer.

    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param is_lane: Ordered ids of py123d lanes.
    :param predecessors: Predecessors of each py123d lane.
    :param successors: Successors of each py123d lane.
    :param openings: Driving neighbour each entry / exit lane opens from or closes into (see collect_topology).
    :return: Lane ids per lane group id, returned once all objects are yielded.
    """

    def boundaries(key: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # NOTE: Entry / exit lanes run at full width along their far boundary (the one away from the driving neighbour
        # they open from / close into, e.g. a shoulder) where the XODR lane is still (already) narrower, like
        # _extend_lane_with_shoulder of the former parser. Elsewhere the XODR boundaries are kept.
        lane = lanes[key]
        inner, outer = lane["inner"].copy(), lane["outer"].copy()
        for neighbour, at_start in openings.get(key, []):
            near, far = (inner, outer) if abs(lanes[neighbour]["id"]) < abs(lane["id"]) else (outer, inner)
            offset = near[:, :2] - far[:, :2]
            width = np.linalg.norm(offset, axis=1)
            full = int(np.argmax(width))
            reached = width >= width[full] - MIN_LANE_WIDTH
            narrow = np.cumsum(reached) == 0 if at_start else np.cumsum(reached[::-1])[::-1] == 0
            tangent = np.gradient(far[:, :2], axis=0)
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
            normal *= np.sign(normal[full] @ offset[full])
            near[narrow, :2] = far[narrow, :2] + width[full] * normal[narrow]
        return inner, outer

    groups = {}
    for key in is_lane:
        groups.setdefault(key.rsplit("_", 1)[0], []).append(key)
    lane_group_of = {key: group_id for group_id, keys in groups.items() for key in keys}

    for group_id, keys in groups.items():
        keys.sort(key=lambda k: abs(lanes[k]["id"]))
        for key in keys:
            lane = lanes[key]
            inner, outer = boundaries(key)

            def neighbour(lane_id: int) -> Optional[str]:
                neighbour_key = lane_key(lane["road"], lane["section"], lane_id)
                return neighbour_key if neighbour_key in is_lane else None

            step = np.sign(lane["id"])
            is_freeway = lane["road_type"] == "motorway" and lane["type"] == "driving"
            yield Lane(
                object_id=key,
                lane_type=LaneType.FREEWAY if is_freeway else LANE_TYPES[lane["type"]],
                left_boundary=Polyline3D.from_array(inner),
                right_boundary=Polyline3D.from_array(outer),
                centerline=Polyline3D.from_array(0.5 * (inner + outer)),
                lane_group_id=group_id,
                left_lane_id=neighbour(lane["id"] - step),
                right_lane_id=neighbour(lane["id"] + step),
                predecessor_ids=predecessors[key],
                successor_ids=successors[key],
                speed_limit_mps=lane["speed"],
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
    return groups
