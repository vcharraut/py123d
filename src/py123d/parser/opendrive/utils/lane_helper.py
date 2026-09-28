import numpy as np

from py123d.datatypes import Lane, LaneGroup, LaneType
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.utils.id_system import lane_key

LANE_TYPES = {"driving": LaneType.SURFACE_STREET, "biking": LaneType.BIKE_LANE}


def iter_lanes_and_lane_groups(lanes, is_lane, predecessors, successors):
    # Lanes and lane groups: one group per road, section and side, lanes ordered inner to outer.
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
    return groups
