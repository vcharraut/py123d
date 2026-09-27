# Quick 2D view of a parsed OpenDRIVE map (pan/zoom in the matplotlib window).
#   source .venv/bin/activate && python scripts/opendrive_parity/viz.py Town07 [xmin ymin xmax ymax] [--save out.png]
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from py123d.datatypes import Carpark, Crosswalk, GenericDrivable, Intersection, Lane, RoadEdge, RoadLine, StopZone, Walkway
from py123d.parser.opendrive.opendrive_map_parser import OpenDriveMapParser

args = [a for a in sys.argv[1:] if a != "--save"]
save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
args = [a for a in args if a != save]
xodr = Path(__file__).parents[2] / f"src/py123d/parser/opendrive/carla_maps/{args[0]}.xodr.gz"
objects = list(OpenDriveMapParser(xodr).iter_map_objects())

fills = {Lane: "#9e9e9e", GenericDrivable: "#d9c89e", Carpark: "#a9c7e8", Walkway: "#9fd49f", Crosswalk: "#ffffff", StopZone: "#e53935"}
fig, ax = plt.subplots(figsize=(12, 12))
for cls, color in fills.items():
    for o in objects:
        if type(o) is cls:
            xy = o.outline.array[:, :2]
            ax.fill(xy[:, 0], xy[:, 1], color=color, lw=0)
for o in objects:
    if isinstance(o, Intersection):
        xy = o.outline.array[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="purple", lw=0.6, ls="--")
    if isinstance(o, RoadLine):
        xy = o.polyline.array[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="#e0b000" if "YELLOW" in o.road_line_type.name else "white", lw=0.7)
    if isinstance(o, RoadEdge):
        xy = o.polyline.array[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="black", lw=1.2, marker=".", ms=2)
if len(args) == 5:
    ax.set_xlim(float(args[1]), float(args[3]))
    ax.set_ylim(float(args[2]), float(args[4]))
ax.set_aspect("equal")
ax.set_facecolor("#f4efe6")
ax.set_title(args[0])
plt.savefig(save, dpi=100, bbox_inches="tight") if save else plt.show()
