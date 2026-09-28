from typing import Any, Dict, Final, Tuple

import numpy as np
import numpy.typing as npt

from py123d.parser.opendrive.xodr_parser.reference import eval_poly, eval_reference

_CHORD_TOLERANCE: Final[float] = 0.01  # [m]
_MAX_INITIAL_STEP: Final[float] = 5.0  # [m]
_MIN_STEP: Final[float] = 0.01  # [m], refinement stops at polynomial discontinuities


def section_at(road: Dict[str, Any], s: float) -> int:
    """Returns the index of the lane section containing s.

    :param road: Parsed road.
    :param s: Position along the road.
    :return: Lane section index.
    """
    return max(k for k, sec in enumerate(road["sections"]) if sec["s"] <= s)


def eval_borders(
    road: Dict[str, Any], section_idx: int, s: npt.NDArray[np.float64]
) -> Dict[int, npt.NDArray[np.float64]]:
    """Evaluates the outer border of every lane of a lane section.

    :param road: Parsed road.
    :param section_idx: Lane section index.
    :param s: Positions along the road, within the lane section.
    :return: (n, 3) outer border per lane id, with lane 0 being the (offset) reference line.
    """
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


def section_grid(
    road: Dict[str, Any], section_idx: int
) -> Tuple[npt.NDArray[np.float64], Dict[int, npt.NDArray[np.float64]]]:
    """Samples a lane section adaptively, until all lane borders deviate less than the chord tolerance.

    :param road: Parsed road.
    :param section_idx: Lane section index.
    :return: s-grid and the lane borders sampled on it (see eval_borders).
    """
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
    counts = np.maximum(np.ceil(np.diff(breaks) / _MAX_INITIAL_STEP).astype(int), 1)
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
        refine = (err > _CHORD_TOLERANCE) & (np.diff(grid) > _MIN_STEP)
        if not np.any(refine):
            return grid, borders
        grid = np.sort(np.concatenate([grid, mid[refine]]))
        borders = eval_borders(road, section_idx, grid)
    return grid, borders
