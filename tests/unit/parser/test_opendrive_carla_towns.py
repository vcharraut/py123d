from pathlib import Path

import pytest
import shapely

from py123d.datatypes import GenericDrivable, StopZone, StopZoneType
from py123d.parser.opendrive import opendrive_map_parser
from py123d.parser.opendrive.opendrive_map_parser import iter_xodr_map_objects

CARLA_MAPS = Path(opendrive_map_parser.__file__).parent / "carla_maps"


@pytest.fixture(scope="module")
def town07():
    return list(iter_xodr_map_objects(CARLA_MAPS / "Town07.xodr.gz"))


class TestTown07:
    def test_every_stop_sign_has_a_stop_zone(self, town07):
        """CARLA's stop test reaches 9 of the 19 stop signs; the other 10 keep their OpenDRIVE stop line."""
        stop_zones = [o for o in town07 if isinstance(o, StopZone) and o.stop_zone_type == StopZoneType.STOP_SIGN]
        assert len(stop_zones) == 19
        assert sum(str(zone.object_id).startswith("carla_") for zone in stop_zones) == 9

    def test_junction_road_is_as_wide_as_carla_paves_it(self, town07):
        """Junction 543 has no shoulder lane in the OpenDRIVE file, CARLA paves it as wide as the roads entering it."""
        beyond_the_turning_lane = shapely.Point(-104.0, -112.0)
        drivables = [shapely.Polygon(o.outline.array[:, :2]) for o in town07 if isinstance(o, GenericDrivable)]
        assert any(drivable.contains(beyond_the_turning_lane) for drivable in drivables)


@pytest.fixture(scope="module")
def town15():
    return list(iter_xodr_map_objects(CARLA_MAPS / "Town15.xodr.gz"))


class TestTown15:
    def test_stop_and_yield_zones_come_from_the_carla_signs(self, town15):
        """Every one of CARLA's 62 signs has a zone; the painted stop lines duplicate traffic-light and yield lines."""
        zones = [o for o in town15 if isinstance(o, StopZone) and o.stop_zone_type != StopZoneType.TRAFFIC_LIGHT]
        assert {str(zone.object_id).split("_")[3] for zone in zones} == {str(n) for n in range(62)}
