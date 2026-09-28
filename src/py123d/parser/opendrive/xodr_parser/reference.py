from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
from scipy.special import fresnel


def eval_poly(records: npt.NDArray[np.float64], s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluates piecewise cubic polynomial records (rows of s_start, a, b, c, d).

    :param records: Polynomial records, sorted by s_start.
    :param s: Positions along the road.
    :return: Polynomial values at s.
    """
    # NOTE: libcarla uses the last record with s_record <= s and yields 0 before the first record.
    if len(records) == 0:
        return np.zeros_like(s)
    idx = np.searchsorted(records[:, 0], s, side="right") - 1
    r = records[np.clip(idx, 0, None)]
    ds = s - r[:, 0]
    return np.where(idx >= 0, r[:, 1] + ds * (r[:, 2] + ds * (r[:, 3] + ds * r[:, 4])), 0.0)


def _odr_spiral(
    s: npt.NDArray[np.float64], c_dot: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    a = np.sqrt(np.pi / np.abs(c_dot))
    fs, fc = fresnel(s / a)
    return a * fc, np.sign(c_dot) * a * fs, 0.5 * c_dot * s * s


def eval_reference(
    road: Dict[str, Any], s: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluates the plan view (reference line) of a road.

    :param road: Parsed road.
    :param s: Positions along the road, clipped to the road length.
    :return: x, y and heading at s.
    """
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
