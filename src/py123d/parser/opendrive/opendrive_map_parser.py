from pathlib import Path
from typing import Iterator, Optional

from py123d.datatypes import BaseMapObject
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.parser.base_dataset_parser import BaseMapParser
from py123d.parser.opendrive.utils.collection import collect_lanes, collect_topology
from py123d.parser.opendrive.utils.lane_helper import iter_lanes_and_lane_groups
from py123d.parser.opendrive.utils.objects_helper import iter_crosswalks
from py123d.parser.opendrive.utils.road_line_helper import iter_road_lines
from py123d.parser.opendrive.utils.signal_helper import collect_stop_lines
from py123d.parser.opendrive.utils.stop_zone_helper import iter_stop_zones
from py123d.parser.opendrive.utils.surface_helper import iter_road_surfaces
from py123d.parser.opendrive.xodr_parser.opendrive import load_xodr


class OpenDriveMapParser(BaseMapParser):
    """Lightweight, picklable handle to one OpenDRIVE map."""

    def __init__(self, xodr_path: Path, location: Optional[str] = None) -> None:
        self._xodr_path = xodr_path
        self._location = location

    def get_map_metadata(self) -> MapMetadata:
        """Returns metadata for this OpenDRIVE map."""
        _location = self._location or self._xodr_path.name.removesuffix("".join(self._xodr_path.suffixes))
        return MapMetadata(dataset="opendrive", location=_location, map_has_z=True, map_is_per_log=False)

    def iter_map_objects(self) -> Iterator[BaseMapObject]:
        """Yields map objects lazily from the XODR file."""
        yield from iter_xodr_map_objects(self._xodr_path)


def iter_xodr_map_objects(xodr_file: Path) -> Iterator[BaseMapObject]:
    """Yields all map objects extracted from an OpenDRIVE (.xodr) file.

    :param xodr_file: Path to the OpenDRIVE (.xodr or .xodr.gz) file.
    """
    roads, junctions, controllers = load_xodr(xodr_file)
    lanes, sampled, raised, curb_lanes = collect_lanes(roads)
    all_successors, all_predecessors, is_lane, predecessors, successors = collect_topology(roads, junctions, lanes)
    groups = yield from iter_lanes_and_lane_groups(lanes, is_lane, predecessors, successors)
    curbs = yield from iter_road_lines(roads, sampled, raised)
    crosswalk_tree, painted_stop_lines, painted_tree, painted_directions = yield from iter_crosswalks(roads)
    stop_lines = collect_stop_lines(roads, junctions, controllers, lanes, all_successors, all_predecessors)
    yield from iter_stop_zones(
        roads,
        lanes,
        is_lane,
        all_successors,
        stop_lines,
        crosswalk_tree,
        painted_stop_lines,
        painted_tree,
        painted_directions,
    )
    yield from iter_road_surfaces(roads, lanes, raised, curb_lanes, curbs, groups)
