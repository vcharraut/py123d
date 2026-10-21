# Counts the lane successor links of a bundled CARLA town whose centerlines are more than 1 cm apart, split into links
# in the OpenDRIVE source and links py123d adds, and whether either lane opens or closes at zero width (widened end).
# Prints, per class, the count, median and max gap [m], and up to 3 examples (lane, successor, gap).
#   python scripts/opendrive_parity/check_lane_gaps.py Town15
import collections, sys
from pathlib import Path
import numpy as np
from py123d.datatypes import Lane
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects
from py123d.parser.opendrive.utils.collection import collect_lanes, collect_topology
from py123d.parser.opendrive.xodr_parser.opendrive import load_xodr

town = sys.argv[1]
path = Path(__file__).parents[2] / "src/py123d/parser/opendrive/carla_maps" / f"{town}.xodr.gz"
roads, junctions, _ = load_xodr(path)
lanes, *_ = collect_lanes(roads)
all_successors, _, _, _, _, openings = collect_topology(roads, junctions, lanes)
objects = {o.object_id: o for o in iter_xodr_map_objects(path) if isinstance(o, Lane)}
gaps, examples = collections.defaultdict(list), collections.defaultdict(list)
for key, lane in objects.items():
    end = lane.centerline_3d.array[-1, :2]
    for successor in lane.successor_ids:
        gap = float(np.linalg.norm(objects[successor].centerline_3d.array[0, :2] - end))
        if gap <= 0.01:
            continue
        cls = "source_link" if successor in all_successors.get(key, []) else "added_link"
        cls += "+widened_end" if successor in openings or key in openings else ""
        gaps[cls].append(gap)
        if len(examples[cls]) < 3:
            examples[cls].append((key, successor, round(gap, 3)))
print(town, sum(map(len, gaps.values())),
      {c: (len(g), round(float(np.median(g)), 3), round(float(np.max(g)), 3)) for c, g in gaps.items()}, dict(examples))
