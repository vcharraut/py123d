import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Final, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt

_SPEED_TO_MPS: Final[Dict[str, float]] = {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704}


def load_xodr(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, ET.Element]]:
    """Loads the roads and junctions of an OpenDRIVE file.

    :param path: Path to the OpenDRIVE (.xodr or .xodr.gz) file.
    :return: Parsed roads and raw junction elements, both keyed by id.
    """
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as f:
        root = ET.parse(f).getroot()
    roads = {int(r.get("id")): _parse_road(r) for r in root.findall("road")}
    junctions = {int(j.get("id")): j for j in root.findall("junction")}
    return roads, junctions


def _poly_records(elements: Iterable[ET.Element], s_key: str = "s", base: float = 0.0) -> npt.NDArray[np.float64]:
    return np.array(
        [[base + float(e.get(s_key)), *(float(e.get(k)) for k in "abcd")] for e in elements], dtype=np.float64
    ).reshape(-1, 5)


def _link_id(link: Optional[ET.Element], tag: str, attribute: str) -> int:
    # NOTE: libcarla keeps only the first link id (0 if missing); road and junction ids share one namespace.
    element = link.find(tag) if link is not None else None
    return int(element.get(attribute)) if element is not None else 0


def _parse_road(road: ET.Element) -> Dict[str, Any]:
    geometry = []
    for g in road.find("planView").findall("geometry"):
        shape = g[0]
        kind = {"line": 0, "arc": 1, "spiral": 2}.get(shape.tag)
        if kind is None:
            raise NotImplementedError(f"Road {road.get('id')}: geometry <{shape.tag}> is not supported")
        c0 = float(shape.get("curvature", shape.get("curvStart", 0.0)))
        c1 = float(shape.get("curvEnd", c0))
        geometry.append([float(g.get(k)) for k in ("s", "x", "y", "hdg", "length")] + [kind, c0, c1])

    if any(float(e.get(k)) != 0.0 for e in road.iter("superelevation") for k in "abcd"):
        raise NotImplementedError(f"Road {road.get('id')}: non-zero superelevation is not supported")

    sections = []
    for sec in road.find("lanes").findall("laneSection"):
        s0 = float(sec.get("s"))
        lanes = {}
        for lane in sec.iter("lane"):
            if lane.find("border") is not None:
                raise NotImplementedError(f"Road {road.get('id')}: lane <border> is not supported")
            link = lane.find("link")
            lanes[int(lane.get("id"))] = {
                "type": lane.get("type"),
                "width": _poly_records(lane.findall("width"), "sOffset", s0),
                "pred": _link_id(link, "predecessor", "id"),
                "succ": _link_id(link, "successor", "id"),
                "marks": [
                    {
                        "s": s0 + float(m.get("sOffset")),
                        "type": m.get("type"),
                        "color": m.get("color"),
                        "lane_change": m.get("laneChange"),
                    }
                    for m in lane.findall("roadMark")
                ],
            }
        sections.append({"s": s0, "lanes": lanes})

    types = []
    for t in road.findall("type"):
        speed = t.find("speed")
        mps = float(speed.get("max")) * _SPEED_TO_MPS[speed.get("unit", "m/s")] if speed is not None else None
        types.append((float(t.get("s")), t.get("type"), mps))

    link = road.find("link")
    elevation = road.find("elevationProfile")
    return {
        "id": int(road.get("id")),
        "length": float(road.get("length")),
        "junction": int(road.get("junction")),
        "pred": _link_id(link, "predecessor", "elementId"),
        "succ": _link_id(link, "successor", "elementId"),
        "geometry": np.array(geometry, dtype=np.float64),
        "elevation": _poly_records(elevation.findall("elevation") if elevation is not None else []),
        "lane_offset": _poly_records(road.find("lanes").findall("laneOffset")),
        "sections": sections,
        "types": types,
        "objects": road.find("objects"),
        "signals": road.find("signals"),
    }
