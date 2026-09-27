import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.special import fresnel

SPEED_TO_MPS = {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704}
CHORD_TOLERANCE = 0.01  # [m]
MAX_INITIAL_STEP = 5.0  # [m]
MIN_STEP = 0.01  # [m], refinement stops at polynomial discontinuities


def load_xodr(path: Path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as f:
        root = ET.parse(f).getroot()
    roads = {int(r.get("id")): parse_road(r) for r in root.findall("road")}
    junctions = {int(j.get("id")): j for j in root.findall("junction")}
    return roads, junctions


def _poly_records(elements, s_key="s", base=0.0):
    return np.array(
        [[base + float(e.get(s_key)), *(float(e.get(k)) for k in "abcd")] for e in elements], dtype=np.float64
    ).reshape(-1, 5)


def _link_id(link, tag, attribute):
    # NOTE: libcarla keeps only the first link id (0 if missing); road and junction ids share one namespace.
    element = link.find(tag) if link is not None else None
    return int(element.get(attribute)) if element is not None else 0


def parse_road(road):
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
        mps = float(speed.get("max")) * SPEED_TO_MPS[speed.get("unit", "m/s")] if speed is not None else None
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


def eval_poly(records, s):
    # NOTE: libcarla uses the last record with s_record <= s and yields 0 before the first record.
    if len(records) == 0:
        return np.zeros_like(s)
    idx = np.searchsorted(records[:, 0], s, side="right") - 1
    r = records[np.clip(idx, 0, None)]
    ds = s - r[:, 0]
    return np.where(idx >= 0, r[:, 1] + ds * (r[:, 2] + ds * (r[:, 3] + ds * r[:, 4])), 0.0)


def _odr_spiral(s, c_dot):
    a = np.sqrt(np.pi / np.abs(c_dot))
    fs, fc = fresnel(s / a)
    return a * fc, np.sign(c_dot) * a * fs, 0.5 * c_dot * s * s


def eval_reference(road, s):
    geo = road["geometry"]
    s = np.clip(s, 0.0, road["length"])
    g = geo[np.clip(np.searchsorted(geo[:, 0], s, side="right") - 1, 0, None)]
    ds = np.clip(s - g[:, 0], 0.0, g[:, 4])
    x0, y0, h0, kind, c0, c1 = g[:, 1], g[:, 2], g[:, 3], g[:, 5], g[:, 6], g[:, 7]

    # line
    x, y, h = x0 + ds * np.cos(h0), y0 + ds * np.sin(h0), h0.copy()

    # arc (also spirals with constant curvature)
    c_dot = np.where(kind == 2, (c1 - c0) / g[:, 4], 0.0)
    arc = (kind == 1) | ((kind == 2) & (c_dot == 0.0) & (c0 != 0.0))
    k = np.where(arc, c0, 1.0)
    x = np.where(arc, x0 + (np.sin(h0 + k * ds) - np.sin(h0)) / k, x)
    y = np.where(arc, y0 - (np.cos(h0 + k * ds) - np.cos(h0)) / k, y)
    h = np.where(arc, h0 + c0 * ds, h)

    # spiral
    spiral = (kind == 2) & (c_dot != 0.0)
    if np.any(spiral):
        cd, s0 = c_dot[spiral], c0[spiral] / c_dot[spiral]
        xa, ya, ta = _odr_spiral(s0, cd)
        xb, yb, tb = _odr_spiral(s0 + ds[spiral], cd)
        dx, dy = xb - xa, yb - ya
        rot = h0[spiral] - ta
        x[spiral] = x0[spiral] + dx * np.cos(rot) - dy * np.sin(rot)
        y[spiral] = y0[spiral] + dx * np.sin(rot) + dy * np.cos(rot)
        h[spiral] = h0[spiral] + tb - ta
    return x, y, h


def section_at(road, s):
    return max(k for k, sec in enumerate(road["sections"]) if sec["s"] <= s)


def eval_borders(road, section_idx, s):
    # Returns {lane_id: (n, 3) outer border}, with lane 0 being the (offset) reference line.
    # NOTE: Records starting at the section end are excluded, so the last sample is the left limit (as in libcarla).
    sections = road["sections"]
    s0 = sections[section_idx]["s"]
    s1 = sections[section_idx + 1]["s"] if section_idx + 1 < len(sections) else road["length"]
    in_section = lambda records: records[(records[:, 0] < s1) | (records[:, 0] <= s0)]
    lanes = sections[section_idx]["lanes"]
    x, y, h = eval_reference(road, s)
    z = eval_poly(in_section(road["elevation"]), s)
    t0 = eval_poly(in_section(road["lane_offset"]), s)
    normal = np.stack([-np.sin(h), np.cos(h)], axis=1)
    base = np.stack([x, y], axis=1)

    t = {0: t0}
    for sign in (1, -1):
        ids = sorted((i for i in lanes if i * sign > 0), key=abs)
        acc = t0
        for i in ids:
            acc = acc + sign * eval_poly(in_section(lanes[i]["width"]), s)
            t[i] = acc
    return {i: np.column_stack([base + normal * ti[:, None], z]) for i, ti in t.items()}


def section_grid(road, section_idx):
    sections = road["sections"]
    s0 = sections[section_idx]["s"]
    s1 = sections[section_idx + 1]["s"] if section_idx + 1 < len(sections) else road["length"]
    lanes = sections[section_idx]["lanes"].values()
    breaks = np.concatenate(
        [
            [s0, s1],
            road["geometry"][:, 0],
            road["elevation"][:, 0],
            road["lane_offset"][:, 0],
            *(lane["width"][:, 0] for lane in lanes),
        ]
    )
    breaks = np.unique(breaks[(breaks >= s0) & (breaks <= s1)])
    counts = np.maximum(np.ceil(np.diff(breaks) / MAX_INITIAL_STEP).astype(int), 1)
    grid = np.unique(
        np.concatenate([np.linspace(a, b, n + 1) for a, b, n in zip(breaks[:-1], breaks[1:], counts)] + [breaks])
    )

    borders = eval_borders(road, section_idx, grid)
    while len(grid) > 1:
        mid = 0.5 * (grid[:-1] + grid[1:])
        mid_borders = eval_borders(road, section_idx, mid)
        err = np.max(
            [np.linalg.norm(mid_borders[i] - 0.5 * (b[:-1] + b[1:]), axis=1) for i, b in borders.items()], axis=0
        )
        refine = (err > CHORD_TOLERANCE) & (np.diff(grid) > MIN_STEP)
        if not np.any(refine):
            return grid, borders
        grid = np.sort(np.concatenate([grid, mid[refine]]))
        borders = eval_borders(road, section_idx, grid)
    return grid, borders
