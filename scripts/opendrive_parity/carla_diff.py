# Compares a py123d OpenDRIVE map with CARLA's rendered top-down capture (capture_carla.py): road IoU, edge distances, diff image.
#   python scripts/opendrive_parity/carla_diff.py Town03 <capture_dir> <tag>   (writes cmp_<town>_<tag>.png into the capture dir)
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from py123d.datatypes import RoadLine, Carpark, GenericDrivable, Intersection, Lane, LaneGroup, RoadEdge, Walkway
from py123d.parser.opendrive.opendrive_map_parser import OpenDriveMapParser

town, cap, tag = sys.argv[1], Path(sys.argv[2]) / sys.argv[1], sys.argv[3]
RES = 0.1
meta = json.loads((cap / "meta.json").read_text())
W, f = meta["W"], meta["W"] / (2 * np.tan(np.radians(meta["FOV"]) / 2))
lab = meta["labels"]
u, v = np.meshgrid(np.arange(W), np.arange(W))
pts, labels = [], []
for t in meta["tiles"]:
    sem = np.frombuffer((cap / f"{t['name']}_semantic_segmentation.bgra").read_bytes(), np.uint8).reshape(W, W, 4)[..., 2]
    d = np.frombuffer((cap / f"{t['name']}_depth.bgra").read_bytes(), np.uint8).reshape(W, W, 4).astype(np.float64)
    depth = (d[..., 2] + d[..., 1] * 256 + d[..., 0] * 65536) / (256**3 - 1) * 1000
    X = t["x"] - depth * (v - W / 2) / f
    Y = t["y"] + depth * (u - W / 2) / f
    c = slice(W // 10, W - W // 10)  # tile centres only (tiles overlap)
    pts.append(np.stack([X[c, c].ravel(), -Y[c, c].ravel()], 1)); labels.append(sem[c, c].ravel())
pts, labels = np.vstack(pts), np.concatenate(labels)
x0, y0 = pts.min(0); x1, y1 = pts.max(0)
shape = (int((y1 - y0) / RES) + 1, int((x1 - x0) / RES) + 1)
ij = ((pts[:, 1] - y0) / RES).astype(int), ((pts[:, 0] - x0) / RES).astype(int)
grid = np.full(shape, 255, np.uint8); grid[ij] = labels
carla_road = np.isin(grid, [lab["Roads"], lab["RoadLines"], lab["Bridge"]])
carla_side = grid == lab["Sidewalks"]
observed = ndimage.binary_closing(grid != 255, iterations=2)
carla_road = ndimage.binary_closing(carla_road, iterations=2) & observed  # fill unobserved pixels between samples
carla_side = ndimage.binary_closing(carla_side, iterations=2) & observed

objs = list(OpenDriveMapParser(Path(__import__("py123d").__file__).parent / f"parser/opendrive/carla_maps/{town}.xodr.gz").iter_map_objects())
to_px = lambda a: [((x - x0) / RES, (y - y0) / RES) for x, y in a[:, :2]]
img = Image.new("L", (shape[1], shape[0])); draw = ImageDraw.Draw(img)
for o in objs:
    if isinstance(o, (Lane, GenericDrivable, Carpark, Intersection)):
        draw.polygon(to_px(o.outline.array), fill=1)
ours = np.array(img, bool) & observed
edges = np.vstack([o.polyline.array[:, :2] for o in objs if isinstance(o, RoadEdge)]) if any(isinstance(o, RoadEdge) for o in objs) else np.zeros((0, 2))
if len(edges):  # densify
    edges = np.vstack([np.linspace(a, b, max(int(np.linalg.norm(b - a) / RES), 1) + 1) for o in objs if isinstance(o, RoadEdge) for a, b in zip(o.polyline.array[:-1, :2], o.polyline.array[1:, :2])])

roi = ndimage.binary_dilation(ours, iterations=100) & ndimage.binary_erosion(observed, iterations=5)
far = (carla_road & ~roi).sum() * RES**2
carla_road &= roi
inter, union = (carla_road & ours).sum(), (carla_road | ours).sum()
boundary = carla_road ^ ndimage.binary_erosion(carla_road)
dist = ndimage.distance_transform_edt(~boundary) * RES
ei = np.clip(((edges[:, 1] - y0) / RES).astype(int), 0, shape[0] - 1), np.clip(((edges[:, 0] - x0) / RES).astype(int), 0, shape[1] - 1)
e_err = dist[ei]
edge_img = np.zeros(shape, bool); edge_img[ei] = True
d_to_ours = ndimage.distance_transform_edt(~edge_img) * RES
b_err = d_to_ours[boundary & roi]
print(f"{town} [{tag}] road IoU {inter / union:.3f} | CARLA-only {(carla_road & ~ours).sum() * RES**2:.0f} m2 | ours-only {(ours & ~carla_road).sum() * RES**2:.0f} m2 | CARLA road >10m from ours (ignored) {far:.0f} m2"
      f" | our edges -> CARLA boundary: median {np.median(e_err):.2f} p90 {np.percentile(e_err, 90):.2f} m, >1m {np.mean(e_err > 1) * 100:.1f}%"
      f" | CARLA boundary -> our edges: median {np.median(b_err):.2f} p90 {np.percentile(b_err, 90):.2f} m, >1m {np.mean(b_err > 1) * 100:.1f}%")
lines = grid == lab["RoadLines"]
line_dist = ndimage.distance_transform_edt(~lines) * RES
lp = np.vstack([o.polyline.array[:, :2] for o in objs if isinstance(o, RoadLine) and "SOLID" in o.road_line_type.name])
li = np.clip(((lp[:, 1] - y0) / RES).astype(int), 0, shape[0] - 1), np.clip(((lp[:, 0] - x0) / RES).astype(int), 0, shape[1] - 1)
ok = roi[li]
print(f"  registration: our solid road-line vertices -> CARLA RoadLines pixels median {np.median(line_dist[li][ok]):.2f} p90 {np.percentile(line_dist[li][ok], 90):.2f} m")
# signed kerb offset: + means our edge lies outside CARLA road
inside = carla_road[ei]
print(f"  our edge vertices inside CARLA road: {inside.mean() * 100:.1f}%")
for name, mask in (("CARLA-only", carla_road & ~ours & roi), ("ours-only", ours & ~carla_road & roi)):
    comp, n = ndimage.label(ndimage.binary_opening(mask, iterations=2))
    sizes = ndimage.sum(mask, comp, range(1, n + 1)) * RES**2
    cents = ndimage.center_of_mass(mask, comp, range(1, n + 1))
    big = sorted(((s, c) for s, c in zip(sizes, cents) if s > 3), key=lambda x: -x[0])
    print(f"  {name} blobs >3m2: {len(big)}:", [(round(s), round(x0 + c[1] * RES), round(y0 + c[0] * RES)) for s, c in big[:12]])
rgb = np.full(shape + (3,), 245, np.uint8)
rgb[~observed] = (255, 255, 0)
rgb[carla_side] = (170, 220, 170)
rgb[carla_road & ours] = (150, 150, 150)
rgb[carla_road & ~ours] = (230, 60, 60)
rgb[ours & ~carla_road] = (60, 90, 230)
rgb[ndimage.binary_dilation(edge_img, iterations=1)] = (0, 0, 0)
Image.fromarray(rgb[::-1]).save(cap / f"cmp_{town}_{tag}.png")
(cap / f"cmp_{town}_{tag}.json").write_text(json.dumps({"x0": float(x0), "y0": float(y0), "h": shape[0], "res": RES}))
