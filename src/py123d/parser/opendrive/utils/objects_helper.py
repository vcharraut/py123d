from typing import Any, Dict, Final, Generator, List, Tuple

import numpy as np
import numpy.typing as npt
import shapely

from py123d.datatypes import BaseMapObject, Crosswalk
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.xodr_parser.lane import eval_borders, section_at
from py123d.parser.opendrive.xodr_parser.reference import eval_poly, eval_reference

_STOP_LINE_SNAP: Final[float] = 0.2  # [m]
_STOP_LINE_SIGNAL_LENGTH: Final[float] = 4.0  # [m], about one lane: centred on a lane (or on the border of two lanes)
_STOP_LINE_SIGNAL_MERGE: Final[float] = 0.5  # [m], stop line signals closer than this are the same painted line


def iter_crosswalks(
    roads: Dict[int, Dict[str, Any]],
) -> Generator[
    BaseMapObject, None, Tuple[shapely.STRtree, List[shapely.Polygon], shapely.STRtree, npt.NDArray[np.float64]]
]:
    """Yields crosswalks (libcarla Map::GetAllCrosswalkZones): outline corners relative to the offset reference line.

    Painted "StopLine*" objects use the same outline convention and are kept to place stop zones, as are the
    "StopLine*" signals (a point across the road).

    :param roads: Parsed roads, keyed by id.
    :return: Crosswalk tree, painted stop line areas, their tree and their (n, 2) unit directions, returned once all
        crosswalks are yielded.
    """

    def frame(
        road: Dict[str, Any], s: float, t: float, hdg: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # NOTE: libcarla's lane 0 transform: heading plus the raw laneOffset derivative, pitch is the raw elevation
        # derivative. The whole local (u, v, z) frame is rotated with that transform.
        s_arr = np.array([s])
        base = eval_borders(road, section_at(road, s), s_arr)[0][0]
        dh = lambda records: np.diff(eval_poly(records, s + np.array([-1e-6, 1e-6])))[0] / 2e-6
        h = eval_reference(road, s_arr)[2][0] + dh(road["lane_offset"])
        a, p = h + hdg, dh(road["elevation"])
        u_axis = np.array([np.cos(p) * np.cos(a), np.cos(p) * np.sin(a), np.sin(p)])
        v_axis = np.array([-np.sin(a), np.cos(a), 0.0])
        pivot = base + t * np.array([-np.cos(p) * np.sin(h), np.cos(p) * np.cos(h), np.sin(p)])
        return pivot, np.stack([u_axis, v_axis, np.cross(u_axis, v_axis)])

    crosswalks, painted_stop_lines, painted_directions, signal_centres = [], [], [], []
    for road_id, road in roads.items():
        for obj in road["objects"].findall("object") if road["objects"] is not None else []:
            has_outline = obj.find("outline") is not None
            is_crosswalk = has_outline and (obj.get("type") == "crosswalk" or "crosswalk" in obj.get("name", "").lower())
            is_stop_line = has_outline and obj.get("name", "").startswith("StopLine")
            if not is_crosswalk and not is_stop_line:
                continue
            corners = np.array(
                [[float(c.get(a)) for a in "uvz"] for c in obj.find("outline").findall("cornerLocal")]
            )
            pivot, axes = frame(road, float(obj.get("s")), float(obj.get("t", 0.0)), float(obj.get("hdg", 0.0)))
            outline = pivot + corners @ axes
            if is_stop_line:
                painted_stop_lines.append(shapely.LineString(outline[:, :2]).buffer(_STOP_LINE_SNAP))
                edges = np.diff(outline[:, :2], axis=0)
                longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
                painted_directions.append(longest / np.linalg.norm(longest))
                continue
            crosswalks.append(shapely.Polygon(outline[:, :2]) if len(outline) > 2 else shapely.Polygon())
            yield Crosswalk(
                object_id=f"{road_id}_object_{obj.get('id')}",
                outline=Polyline3D.from_array(np.vstack([outline, outline[:1]])),
            )

        for signal in road["signals"] if road["signals"] is not None else []:
            if signal.tag != "signal" or not (signal.get("name") or "").startswith("StopLine"):
                continue
            # NOTE: A stop line signal is a point; the line runs across the road (hOffset), one lane long.
            s, t, hdg = (float(signal.get(k, 0.0)) for k in ("s", "t", "hOffset"))
            pivot, axes = frame(road, s, t, hdg)
            if any(np.linalg.norm(pivot[:2] - c) < _STOP_LINE_SIGNAL_MERGE for c in signal_centres):
                continue
            signal_centres.append(pivot[:2])
            ends = pivot[:2] + np.outer([-0.5, 0.5], _STOP_LINE_SIGNAL_LENGTH * axes[0, :2])
            painted_stop_lines.append(shapely.LineString(ends).buffer(_STOP_LINE_SNAP))
            painted_directions.append(axes[0, :2] / np.linalg.norm(axes[0, :2]))
    crosswalk_tree = shapely.STRtree([c if c.is_valid else shapely.make_valid(c) for c in crosswalks])
    painted_tree = shapely.STRtree(painted_stop_lines)
    painted_directions = np.array(painted_directions).reshape(-1, 2)
    return crosswalk_tree, painted_stop_lines, painted_tree, painted_directions
