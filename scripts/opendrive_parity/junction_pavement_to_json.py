# Turns the junction surface grids of export_junction_pavement.py (<dir>/<Town>.json) into pavement outlines and writes
# them to src/py123d/parser/opendrive/carla_maps/junction_pavement.json as {town: [[z, exterior, hole, ...], ...]}: the
# paved cells reached from the junction's lanes without a height step (kerb), smoothed, as polygons of (x, y) rings with the
# height at their representative point.
#   python scripts/opendrive_parity/junction_pavement_to_json.py <dir>
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

KERB, MIN_HOLE, MIN_AREA, TOLERANCE, SMOOTH = 0.06, 2.0, 1.0, 0.1, 0.4  # [m], [m^2], [m^2], [m], [m]
out = Path(__file__).parents[2] / "src/py123d/parser/opendrive/carla_maps/junction_pavement.json"
towns = json.loads(out.read_text()) if out.exists() else {}

for path in sorted(Path(sys.argv[1]).glob("Town*.json")):
    rows = []
    for junction in json.loads(path.read_text()):
        nx, ny, sub, step = junction["nx"], junction["ny"], junction["sub"], junction["step"] / junction["sub"]
        paved = np.array(list(junction["paved"])).reshape(ny, nx) == "1"
        height = np.array(junction["z"], dtype=np.float64).reshape(ny, nx)
        paved, height = (np.repeat(np.repeat(a, sub, axis=0), sub, axis=1) for a in (paved, height))
        for key, (bits, z) in junction["fine"].items():
            i, j = (int(v) * sub for v in key.split(","))
            paved[j : j + sub, i : i + sub] = (np.array(list(bits)) == "1").reshape(sub, sub)
            height[j : j + sub, i : i + sub] = np.array(z, dtype=np.float64).reshape(sub, sub)

        # NOTE: Neighbouring paved cells belong together unless a kerb-high step separates them.
        index = np.arange(paved.size).reshape(paved.shape)
        pairs = []
        for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
            joined = paved[a] & paved[b] & (np.abs(height[a] - height[b]) < KERB)
            pairs.append(np.stack([index[a][joined], index[b][joined]]))
        pairs = np.hstack(pairs)
        graph = coo_matrix((np.ones(pairs.shape[1], dtype=bool), (pairs[0], pairs[1])), shape=(paved.size, paved.size))
        labels = connected_components(graph, directed=False)[1].reshape(paved.shape)
        seeds = np.array(junction["seeds"])
        si = np.clip(((seeds[:, 0] - junction["x0"]) / step).astype(int), 0, paved.shape[1] - 1)
        sj = np.clip(((seeds[:, 1] - junction["y0"]) / step).astype(int), 0, paved.shape[0] - 1)
        mask = paved & np.isin(labels, np.unique(labels[sj, si][paved[sj, si]]))
        # NOTE: Where the road mesh lies in the plane of another surface the rays hit either one: opening and closing
        # remove the speckles, the strips reached through them and the pinholes this leaves along the pavement border.
        r = round(SMOOTH / step)
        disk = np.hypot(*np.mgrid[-r : r + 1, -r : r + 1]) <= r
        mask = ndimage.binary_closing(ndimage.binary_opening(mask, disk), disk)

        # One box per run of cells in a row.
        edges = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
        (r0, c0), (_, c1) = np.nonzero(edges == 1), np.nonzero(edges == -1)
        x, y = junction["x0"] + np.stack([c0, c1]) * step, junction["y0"] + np.stack([r0, r0 + 1]) * step
        area = shapely.union_all(shapely.box(x[0], y[0], x[1], y[1]), grid_size=0.001)
        for polygon in shapely.get_parts(area):
            holes = [r for r in polygon.interiors if shapely.Polygon(r).area >= MIN_HOLE]
            polygon = shapely.set_precision(shapely.Polygon(polygon.exterior, holes).simplify(TOLERANCE), 0.01)
            if polygon.geom_type != "Polygon" or polygon.area < MIN_AREA:
                continue
            point = polygon.representative_point()
            i, j = int((point.x - junction["x0"]) / step), int((point.y - junction["y0"]) / step)
            rings = [np.round(shapely.get_coordinates(r), 2).tolist() for r in [polygon.exterior, *polygon.interiors]]
            rows.append([round(float(height[j, i]), 2), *rings])
    towns[path.stem] = rows
    print(path.stem, len(rows), "polygons", round(sum(shapely.Polygon(r[1], r[2:]).area for r in rows)), "m2")
out.write_text(json.dumps(dict(sorted(towns.items())), separators=(",", ":")))
