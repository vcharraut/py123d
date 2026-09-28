import numpy as np
import shapely

from py123d.datatypes import Crosswalk
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.xodr_parser.lane import eval_borders, section_at
from py123d.parser.opendrive.xodr_parser.reference import eval_poly, eval_reference

STOP_LINE_SNAP = 0.2  # [m]


def iter_crosswalks(roads):
    # Crosswalks (libcarla Map::GetAllCrosswalkZones): outline corners relative to the offset reference line.
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
    return crosswalk_tree, painted_stop_lines, painted_tree, painted_directions
