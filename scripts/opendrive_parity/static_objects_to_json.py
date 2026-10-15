# Turns the solid intervals of export_obstacles.py (<dir>/<Town>.json, <dir>/<Town>.boxes.json) into static objects
# and writes them to src/py123d/parser/opendrive/carla_maps/static_objects.json as {town: [[id, type, height,
# footprint], ...]}: one footprint box (4 corners of x, y, ground z) per connected piece of a CARLA object, cut into
# CELL pieces so that a curved wall never boxes in the road.
#   python scripts/opendrive_parity/static_objects_to_json.py <dir>
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely

STEP, CLOSING, CELL, MIN_AREA, BOX_TOLERANCE, BAND = 0.1, 0.25, 5.0, 0.001, 0.1, 2.0  # [m], [m], [m], [m^2], [m], [m]
PROBES = (0.3, 1.4)  # [m], lowest and highest ray above ground
TYPES = {"GuardRail": "BARRIER", "Fences": "BARRIER", "Walls": "BARRIER", "Poles": "POLE", "TrafficSigns": "POLE",
         "TrafficLight": "POLE", "Car": "VEHICLE", "Truck": "VEHICLE", "Bus": "VEHICLE", "Motorcycle": "VEHICLE",
         "Bicycle": "VEHICLE", "Vegetation": "VEGETATION", "Buildings": "BUILDING", "Bridge": "BUILDING"}
out = Path(__file__).parents[2] / "src/py123d/parser/opendrive/carla_maps/static_objects.json"
towns = json.loads(out.read_text()) if out.exists() else {}


def cells(geometry):
    x0, y0, x1, y1 = geometry.bounds
    grid = [shapely.box(x, y, x + CELL, y + CELL) for x in np.arange(x0, x1, CELL) for y in np.arange(y0, y1, CELL)]
    pieces = shapely.get_parts(shapely.intersection(geometry, grid))
    return [p for p in pieces if p.geom_type == "Polygon" and p.area >= MIN_AREA]


for path in sorted(p for p in Path(sys.argv[1]).glob("Town*.json") if not p.name.endswith(".boxes.json")):
    boxes = json.loads(path.with_suffix(".boxes.json").read_text())
    names = sorted(boxes)
    polygons = [shapely.MultiPoint(boxes[n][0]).convex_hull.buffer(BOX_TOLERANCE, join_style="mitre") for n in names]
    areas, tree = shapely.area(polygons), shapely.STRtree(polygons)
    by_finder = defaultdict(list)
    for row in json.loads(path.read_text()):
        by_finder[row[5]].append(row)
    # NOTE: A ray across an object also hits its neighbours, so the hits are sorted by bounding box: every box at the
    # probed height claims the hits inside it, smaller boxes first (a lamp post's box spans its arms and the drums
    # below them).
    claims, leftovers = defaultdict(list), defaultdict(list)
    for finder, rows in by_finder.items():
        r = np.array([row[:5] for row in rows], dtype=float)
        z = float(np.median(r[:, 4]))
        strips = shapely.buffer(shapely.linestrings(r[:, :4].reshape(-1, 2, 2)), STEP / 2, cap_style="flat")
        strips = np.where(shapely.area(strips) > 0, strips, shapely.buffer(shapely.points(r[:, :2]), STEP / 2))
        rest = shapely.union_all(strips, grid_size=0.01).buffer(CLOSING).buffer(-CLOSING)
        candidates = [
            i
            for i in tree.query(rest, predicate="intersects")
            if boxes[names[i]][2] <= z + PROBES[1] and boxes[names[i]][3] >= z + PROBES[0]
        ]
        for i in sorted(candidates, key=lambda i: areas[i]):
            piece = rest.intersection(polygons[i])
            if piece.area >= MIN_AREA:
                claims[i].append((piece, z))
                rest = rest.difference(polygons[i])
        if rest.area >= MIN_AREA:
            leftovers[np.floor(z / BAND)].append((rest, z, rows[0][6], max(row[7] for row in rows), finder))
    objects = []
    for i, pieces in claims.items():
        z = max(pieces, key=lambda p: p[0].area)[1]
        name, (_, label, _, top) = names[i], boxes[names[i]]
        geometry = shapely.union_all([p for p, _ in pieces])
        for k, piece in enumerate(cells(geometry)):
            corners = shapely.get_coordinates(shapely.minimum_rotated_rectangle(piece))[:4]
            box = np.round(np.c_[corners, np.full(4, z)], 2)
            objects.append([f"{name}_{k}", TYPES.get(label, "OTHER"), round(top - z, 2), box.tolist()])
    # NOTE: Hits outside every box (meshes without an own bounding box, e.g. bridge parts) keep their finder's label.
    for band, parts in leftovers.items():
        finders = shapely.STRtree([p for p, *_ in parts])
        for n, piece in enumerate(cells(shapely.union_all([p for p, *_ in parts]))):
            _, z, label, height, finder = parts[int(finders.query(piece, predicate="intersects")[0])]
            corners = shapely.get_coordinates(shapely.minimum_rotated_rectangle(piece))[:4]
            box = np.round(np.c_[corners, np.full(4, z)], 2)
            objects.append([f"{finder}_rest_{n}", TYPES.get(label, "OTHER"), round(height, 2), box.tolist()])
    towns[path.stem] = objects
    print(path.stem, len(by_finder), "scanned objects,", len(claims), "boxes with hits ->", len(objects), "objects")
out.write_text(json.dumps(towns, separators=(",", ":")))
