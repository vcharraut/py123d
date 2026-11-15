from pathlib import Path

import pytest
import shapely

from py123d.datatypes import GenericDrivable
from py123d.parser.opendrive import opendrive_map_parser
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

CARLA_MAPS = Path(opendrive_map_parser.__file__).parent / "carla_maps"


@pytest.fixture(scope="module")
def town07():
    return list(iter_xodr_map_objects(CARLA_MAPS / "Town07.xodr.gz"))


class TestTown07:
    def test_junction_road_is_as_wide_as_carla_paves_it(self, town07):
        """Junction 543 has no shoulder lane in the OpenDRIVE file, CARLA paves it as wide as the roads entering it."""
        beyond_the_turning_lane = shapely.Point(-104.0, -112.0)
        drivables = [shapely.Polygon(o.outline.array[:, :2]) for o in town07 if isinstance(o, GenericDrivable)]
        assert any(drivable.contains(beyond_the_turning_lane) for drivable in drivables)

