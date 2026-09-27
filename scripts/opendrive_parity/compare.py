# Compares the py123d OpenDRIVE parser against the libcarla oracle dumped by dump_oracle.py.
#   source .venv/bin/activate && python scripts/opendrive_parity/compare.py <oracle_dir> [Town01 Town03 ...]
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree

from py123d.datatypes import Crosswalk, Lane, RoadLine, StopZone
from py123d.parser.opendrive.opendrive_map_parser import OpenDriveMapParser
from py123d.parser.opendrive.xodr import load_xodr

MAPS_DIR = Path(__file__).parents[2] / "src/py123d/parser/opendrive/carla_maps"

oracle_dir = Path(sys.argv[1])
towns = sys.argv[2:] or sorted(p.name.split(".")[0] for p in oracle_dir.glob("*.json.gz"))
lane_key = lambda road, section, lane: f"{road}_{section}_{'left' if lane > 0 else 'right'}_{lane}"
stats = lambda e: f"mean {np.mean(e):.4f} p99 {np.percentile(e, 99):.4f} max {np.max(e):.4f}" if len(e) else "n/a"

for town in towns:
    oracle = json.load(gzip.open(oracle_dir / f"{town}.json.gz"))
    smp = {k: np.array(v) for k, v in oracle["samples"].items()}
    objects = list(OpenDriveMapParser(MAPS_DIR / f"{town}.xodr.gz").iter_map_objects())
    lanes = {o.object_id: o for o in objects if isinstance(o, Lane)}

    driving = np.isin(smp["lane_type"], ["Driving", "Biking"])
    keys = np.array([lane_key(r, s, l) for r, s, l in zip(smp["road"], smp["section"], smp["lane"])])
    xy, z, w = np.stack([smp["x"], smp["y"]], axis=1), smp["z"], smp["width"]

    oracle_lanes = set(keys[driving])
    print(f"== {town}: oracle driving lanes {len(oracle_lanes)}, py123d lanes {len(lanes)}")
    print(f"  missing {sorted(oracle_lanes - lanes.keys())[:5]}, extra {sorted(lanes.keys() - oracle_lanes)[:5]}")

    err = defaultdict(list)
    for key in oracle_lanes & lanes.keys():
        idx = np.nonzero(keys == key)[0]
        lane, c = lanes[key], xy[idx]
        # NOTE: libcarla's lane yaw includes width derivatives, so borders are compared through the width.
        # Boundaries share the centerline's s-grid, so they are interpolated at the same segment and fraction.
        cl, lb, rb = lane.centerline.array, lane.left_boundary.array, lane.right_boundary.array
        a, d = cl[:-1, :2], cl[1:, :2] - cl[:-1, :2]
        u = np.clip(np.einsum("psk,sk->ps", c[:, None] - a[None], d) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
        dist = np.linalg.norm(a[None] + u[..., None] * d[None] - c[:, None], axis=2)
        seg = dist.argmin(1)
        f = u[np.arange(len(c)), seg][:, None]
        at = lambda arr: arr[seg] + f * (arr[seg + 1] - arr[seg])
        err["centre"].extend(dist.min(1))
        err["width"].extend(np.abs(np.linalg.norm(at(lb)[:, :2] - at(rb)[:, :2], axis=1) - w[idx]))
        err["z"].extend(np.abs(at(cl)[:, 2] - z[idx]))
    for name in ("centre", "width", "z"):
        print(f"  {name:6s} [m] {stats(err[name])}")

    roads, _ = load_xodr(MAPS_DIR / f"{town}.xodr.gz")
    succ = defaultdict(set)
    for a, b in oracle["topology"]:
        # NOTE: GetTopology pairs dead-end lanes with a waypoint at their own end, skip those.
        link = roads[a[0]]["sections"][a[1]]["lanes"][a[2]]["succ" if a[2] <= 0 else "pred"]
        if not (a[0] == b[0] and a[2] == b[2] and link == 0):
            succ[lane_key(*a)].add(lane_key(*b))
    # NOTE: libcarla only exposes topology and neighbours for driving lanes.
    common = set(keys[smp["lane_type"] == "Driving"]) & lanes.keys()
    bad_succ = [k for k in common if succ[k] != set(lanes[k].successor_ids)]
    print(f"  successors mismatch {len(bad_succ)}/{len(common)} e.g. {[(k, sorted(succ[k]), lanes[k].successor_ids) for k in bad_succ[:3]]}")

    neighbours = {}
    for k, r, s, k_lane, lft, rgt in zip(*(a[driving] for a in (keys, smp["road"], smp["section"], smp["lane"], smp["left_lane"], smp["right_lane"]))):
        to_key = lambda n: lane_key(r, s, n) if n * k_lane > 0 and lane_key(r, s, n) in oracle_lanes else None
        neighbours[k] = (to_key(lft), to_key(rgt))
    bad_nb = [k for k in common if neighbours[k] != (lanes[k].left_lane_id, lanes[k].right_lane_id)]
    print(f"  neighbours mismatch {len(bad_nb)}/{len(common)} e.g. {[(k, neighbours[k], (lanes[k].left_lane_id, lanes[k].right_lane_id)) for k in bad_nb[:3]]}")

    ours_cw = np.vstack([o.outline.array for o in objects if isinstance(o, Crosswalk)] or [np.zeros((0, 3))])
    oracle_cw = np.array(oracle["crosswalks"]).reshape(-1, 3)
    cw_err = cKDTree(ours_cw).query(oracle_cw)[0] if len(ours_cw) else []
    print(f"  crosswalk corners oracle {len(oracle_cw)} py123d {len(ours_cw)}, dist 3D [m] {stats(cw_err)}")
    print(f"  road lines {sum(isinstance(o, RoadLine) for o in objects)}, stop zones {sum(isinstance(o, StopZone) for o in objects)}")
