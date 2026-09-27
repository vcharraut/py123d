# Dumps libcarla's interpretation of the bundled CARLA .xodr maps (ground truth for the py123d OpenDRIVE parser).
# Needs Python 3.12 + the CARLA wheel, no server:
#   unzip <CARLA>/PythonAPI/carla/dist/carla-0.9.16-cp312-*.whl -d <whl_dir>
#   PYTHONPATH=<whl_dir> python3.12 scripts/opendrive_parity/dump_oracle.py <out_dir> [Town01 Town03 ...]
# Output is in the xodr frame (CARLA's Unreal y and yaw are flipped back).
import gzip
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import carla

MAPS_DIR = Path(__file__).parents[2] / "src/py123d/parser/opendrive/carla_maps"
STEP = 1.0
EPS = 1e-3

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
towns = sys.argv[2:] or sorted(p.name.split(".")[0] for p in MAPS_DIR.glob("*.xodr.gz"))

for town in towns:
    xodr = gzip.open(MAPS_DIR / f"{town}.xodr.gz").read().decode()
    cmap = carla.Map(town, xodr)
    root = ET.fromstring(xodr)

    rows = []
    for road in root.iter("road"):
        road_id, length = int(road.get("id")), float(road.get("length"))
        sections = road.find("lanes").findall("laneSection")
        starts = [float(sec.get("s")) for sec in sections] + [length]
        for k, sec in enumerate(sections):
            s0, s1 = starts[k] + EPS, starts[k + 1] - EPS
            if s1 <= s0:
                continue
            s_grid = [s0 + i * STEP for i in range(math.ceil((s1 - s0) / STEP))] + [s1]
            lane_ids = [int(lane.get("id")) for lane in sec.iter("lane") if lane.get("id") != "0"]
            for lane_id in lane_ids:
                for s in s_grid:
                    wp = cmap.get_waypoint_xodr(road_id, lane_id, s)
                    if wp is None:
                        rows.append((road_id, k, lane_id, s, *[math.nan] * 5, "MISSING", "", "", "", -1, 0, 0))
                        continue
                    loc, yaw = wp.transform.location, math.radians(wp.transform.rotation.yaw)
                    is_driving = wp.lane_type == carla.LaneType.Driving
                    left = wp.get_left_lane() if is_driving else None
                    right = wp.get_right_lane() if is_driving else None
                    rows.append(
                        (
                            road_id,
                            wp.section_id,
                            lane_id,
                            s,
                            loc.x,
                            -loc.y,
                            loc.z,
                            -yaw,
                            wp.lane_width,
                            str(wp.lane_type),
                            str(wp.left_lane_marking.type),
                            str(wp.right_lane_marking.type),
                            str(wp.lane_change),
                            wp.junction_id if wp.is_junction else -1,
                            left.lane_id if left is not None and left.road_id == road_id else 0,
                            right.lane_id if right is not None and right.road_id == road_id else 0,
                        )
                    )

    names = "road section lane s x y z yaw width lane_type left_mark right_mark lane_change junction left_lane right_lane"
    samples = dict(zip(names.split(), map(list, zip(*rows))))

    key = lambda wp: [wp.road_id, wp.section_id, wp.lane_id]
    topology = [[key(a), key(b)] for a, b in cmap.get_topology()]

    # NOTE: libcarla returns all crosswalk corners flattened, outlines are not always closed at their first corner.
    crosswalks = [[loc.x, -loc.y, loc.z] for loc in cmap.get_crosswalks()]

    landmarks = [
        {
            "id": lm.id,
            "name": lm.name,
            "type": lm.type,
            "sub_type": lm.sub_type,
            "road": lm.road_id,
            "s": lm.s,
            "t": lm.t,
            "value": lm.value,
            "xyz": [lm.transform.location.x, -lm.transform.location.y, lm.transform.location.z],
            "validities": [list(v) for v in lm.get_lane_validities()],
        }
        for lm in cmap.get_all_landmarks()
    ]

    with gzip.open(out_dir / f"{town}.json.gz", "wt") as f:
        json.dump({"samples": samples, "topology": topology, "crosswalks": crosswalks, "landmarks": landmarks}, f)
    print(town, "samples", len(rows), "topology", len(topology), "crosswalks", len(crosswalks), "landmarks", len(landmarks))
