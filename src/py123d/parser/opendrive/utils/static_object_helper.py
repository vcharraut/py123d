import json
from pathlib import Path
from typing import Any, Final, Iterator, List, Optional

import numpy as np

from py123d.datatypes import StaticObject, StaticObjectType
from py123d.geometry.polyline import Polyline3D

# NOTE: Exported from CARLA 0.9.16 (scripts/opendrive_parity/export_obstacles.py, static_objects_to_json.py): the
# objects vehicles collide with near the drivable lanes (guard rails, poles, crash barrels, walls, parked cars, ...),
# as [id, type, height, footprint box corners (x, y, ground z)], measured at 0.3-1.4 m above ground.
_STATIC_OBJECTS_FILE: Final[Path] = Path(__file__).parents[1] / "carla_maps" / "static_objects.json"


def load_static_objects(town: Optional[str]) -> Optional[List[List[Any]]]:
    """Loads the static objects CARLA places on a town map.

    :param town: Town name, i.e. the OpenDRIVE file name without suffixes.
    :return: Static objects (see _STATIC_OBJECTS_FILE), None for maps without exported objects (no CARLA town).
    """
    return json.loads(_STATIC_OBJECTS_FILE.read_text()).get(town)


def iter_static_objects(objects: Optional[List[List[Any]]]) -> Iterator[StaticObject]:
    """Yields the static objects of a CARLA town.

    :param objects: Static objects (see load_static_objects).
    """
    for object_id, kind, height, box in objects or []:
        yield StaticObject(
            object_id=object_id,
            outline=Polyline3D.from_array(np.array(box + box[:1])),
            height=height,
            static_object_type=StaticObjectType[kind],
        )
