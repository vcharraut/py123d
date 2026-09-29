from typing import Any, Callable, Dict, Final, Iterator, List, Set, Type

import numpy as np
import numpy.typing as npt
import shapely
from scipy.spatial import cKDTree

from py123d.datatypes import (
    BaseMapObject,
    Carpark,
    GenericDrivable,
    Intersection,
    IntersectionType,
    RoadEdge,
    RoadEdgeType,
    Walkway,
)
from py123d.geometry.polyline import Polyline3D
from py123d.parser.opendrive.utils.collection import KERB_MAX_WIDTH
from py123d.parser.utils.map_utils.road_edge.road_edge_2d_utils import split_line_geometry_by_max_length

_SURFACE_TYPES: Final[Dict[str, Type[BaseMapObject]]] = {
    "parking": Carpark,
    "shoulder": GenericDrivable,
    "border": GenericDrivable,
    "bidirectional": GenericDrivable,
    "restricted": GenericDrivable,
    "sidewalk": Walkway,
}
_KEEP_HOLE_RADIUS: Final[float] = 12.0  # [m], larger holes in the road surface are off-road areas (blocks, roundabouts)
_ISLAND_CURB_FRACTION: Final[float] = 0.05  # holes with this much curb on their boundary are raised islands ...
_ISLAND_MIN_RADIUS: Final[float] = 1.0  # [m] ... unless they are thin slivers
_LAYER_DZ: Final[float] = 1.0  # [m], touching surfaces with larger height difference are on different layers
_MAX_ROAD_EDGE_LENGTH: Final[float] = 100.0  # [m]
_CLOSING: Final[float] = 0.05  # [m], closes slits between adjacent lanes
_GAP_CLOSING: Final[float] = 0.5  # [m]
_MIN_POCKET_AREA: Final[float] = 0.01  # [m^2], smaller kerb pockets are cracks between consecutive kerbs
_SEAL: Final[float] = 0.001  # [m]
_MIN_WALKWAY_ON_ROAD: Final[float] = 1.0  # [m^2], walkways overlapping the road more are clipped, smaller pieces go
_ISLAND_KERB_FRACTION: Final[float] = 0.9  # kerb pockets with this much kerb on their boundary are raised islands
# NOTE: Smaller disconnected road surface pieces are dropped (and thin ones without driving lanes).
_MIN_SURFACE_AREA: Final[float] = 10.0  # [m^2]


def iter_road_surfaces(
    roads: Dict[int, Dict[str, Any]],
    lanes: Dict[str, Dict[str, Any]],
    raised: Set[str],
    curb_lanes: Set[str],
    curbs: List[shapely.LineString],
    groups: Dict[str, List[str]],
) -> Iterator[BaseMapObject]:
    """Yields walkways, other lane surfaces, road edges, generic drivables and intersections.

    The road surface is made of all lanes but sidewalks, raised medians/islands and narrow junction "none" lanes next
    to the reference line (curb strips, checked against CARLA renders). Surfaces above others (bridges) form an upper
    layer. Holes are filled (junction gaps, slivers) unless they are large off-road areas, contain sidewalks or raised
    lanes, or are curbed islands.

    :param roads: Parsed roads, keyed by id.
    :param lanes: Lanes, keyed by lane id (see collect_lanes).
    :param raised: Ids of raised lanes.
    :param curb_lanes: Ids of curb lanes.
    :param curbs: 2D curb lines (see iter_road_lines).
    :param groups: Lane ids per lane group id (see iter_lanes_and_lane_groups).
    """

    def is_curb_strip(lane: Dict[str, Any]) -> bool:
        return (
            lane["type"] == "none"
            and lane["junction"] != -1
            and abs(lane["id"]) == 1
            and np.max(np.linalg.norm(lane["outer"][:, :2] - lane["inner"][:, :2], axis=1)) <= 1.0
        )

    surface_keys = [
        k
        for k, lane in lanes.items()
        if lane["type"] != "sidewalk" and k not in raised and not is_curb_strip(lane) and len(lane["inner"]) > 1
    ]
    outline_of = lambda lane: shapely.Polygon(np.vstack([lane["inner"], lane["outer"][::-1]]))
    surfaces = [outline_of(lanes[k]) for k in surface_keys]
    surfaces = [
        p if p.is_valid else max(shapely.get_parts(shapely.make_valid(p)), key=lambda q: q.area) for p in surfaces
    ]
    blocking = shapely.STRtree(
        [
            shapely.force_2d(outline_of(lane))
            for k, lane in lanes.items()
            if (lane["type"] == "sidewalk" or (k in raised and k not in curb_lanes)) and len(lane["inner"]) > 1
        ]
    )
    blocking_area = shapely.union_all(shapely.make_valid(blocking.geometries))
    kerb_area = shapely.union_all(
        [
            shapely.force_2d(outline_of(lanes[k])).buffer(0)
            for k in curb_lanes
            if lanes[k]["type"] != "sidewalk" and len(lanes[k]["inner"]) > 1
        ]
    )
    kerb_zone = kerb_area.buffer(0.02)
    curb_zone = shapely.union_all(shapely.buffer(curbs, 0.2))
    junction_zone = shapely.union_all(
        shapely.buffer([shapely.force_2d(p) for k, p in zip(surface_keys, surfaces) if lanes[k]["junction"] != -1], 0.2)
    )

    def keep(hole: shapely.Polygon) -> bool:
        radius = shapely.maximum_inscribed_circle(hole, 0.1).length
        curbed = hole.exterior.intersection(curb_zone).length >= _ISLAND_CURB_FRACTION * hole.exterior.length
        blocked = len(blocking.query(hole.buffer(-0.1), predicate="intersects")) > 0
        # NOTE: Holes enclosed by regular roads (not junction gaps) are islands, e.g. roundabout centres.
        in_junction = hole.exterior.intersection(junction_zone).length >= 0.5 * hole.exterior.length
        return radius >= _KEEP_HOLE_RADIUS or blocked or (radius >= _ISLAND_MIN_RADIUS and (curbed or not in_junction))

    def keep_pocket(pocket: shapely.Polygon) -> bool:
        # NOTE: Kerb pockets are road (junction corners, gore areas) unless kerbs enclose them (raised islands) or they
        # are large or blocked off-road areas (checked against CARLA renders).
        radius = shapely.maximum_inscribed_circle(pocket, 0.1).length
        blocked = len(blocking.query(pocket.buffer(-0.1), predicate="intersects")) > 0
        enclosed = pocket.exterior.intersection(kerb_zone).length >= _ISLAND_KERB_FRACTION * pocket.exterior.length
        return radius >= _KEEP_HOLE_RADIUS or blocked or (enclosed and keep(pocket))

    def z_near(polygon: shapely.Polygon, xy: npt.NDArray[np.float64]) -> float:
        coordinates = shapely.get_coordinates(polygon.exterior, include_z=True)
        return coordinates[np.argmin(np.linalg.norm(shapely.get_coordinates(polygon.exterior) - xy, axis=1)), 2]

    upper = set()
    for i, j in zip(*shapely.STRtree(surfaces).query(surfaces, predicate="overlaps")):
        overlap = shapely.intersection(surfaces[i], surfaces[j])
        if i < j and overlap.area > 0.5:
            c = np.array(overlap.centroid.coords[0][:2])
            dz = z_near(surfaces[i], c) - z_near(surfaces[j], c)
            if abs(dz) > _LAYER_DZ:
                upper.add(i if dz > 0 else j)

    def lifter(
        polygons: List[shapely.Polygon],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        # Heights come from the nearest lane vertex of the same layer.
        points = np.vstack(
            [shapely.get_coordinates(p.exterior, include_z=True) for p in polygons] or [np.zeros((0, 3))]
        )
        kd = cKDTree(points[:, :2]) if len(points) else None
        return lambda xy: np.column_stack([xy, points[kd.query(xy)[1], 2]])

    def closed_union(polygons: List[shapely.Polygon]) -> shapely.Geometry:
        union = shapely.union_all(shapely.force_2d(polygons), grid_size=0.001)
        return union.buffer(_CLOSING, join_style="mitre").buffer(-_CLOSING, join_style="mitre")

    driving_tree = shapely.STRtree(
        [
            shapely.force_2d(outline_of(lane)).buffer(-0.05)
            for lane in lanes.values()
            if lane["type"] == "driving" and len(lane["inner"]) > 1
        ]
    )

    def is_thin(piece: shapely.Polygon) -> bool:
        no_driving = not driving_tree.query(piece, predicate="intersects").size
        return no_driving and shapely.maximum_inscribed_circle(piece, 0.05).length < 1.0

    regions, lifts, fillers, dropped = [], [], [], []
    for layer in ([s for n, s in enumerate(surfaces) if n not in upper], [surfaces[n] for n in upper]):
        lift = lifter(layer)
        parts = shapely.get_parts(closed_union(layer))
        # NOTE: Tiny pieces and thin pieces without any driving lane (shoulder rings around islands) are not road.
        small = [p for p in parts if p.area < _MIN_SURFACE_AREA or is_thin(p)]
        dropped += small
        union = shapely.union_all([p for p in parts if not any(p is q for q in small)])
        # NOTE: Gaps narrower than 2 * _GAP_CLOSING between road surfaces (junction wedges) are road, never over
        # sidewalks or raised lanes.
        gaps = union.buffer(_GAP_CLOSING, join_style="mitre").buffer(-_GAP_CLOSING, join_style="mitre")
        union = shapely.union_all([union, gaps.difference(blocking_area)])
        holes = [shapely.Polygon(r) for p in shapely.get_parts(union) for r in p.interiors]
        # NOTE: Ground-level holes containing kerbs are decided as kerb pockets (below).
        filled = [
            h for h in holes if not keep(h) and (regions or h.intersection(kerb_area).area <= _MIN_POCKET_AREA)
        ]
        region = shapely.union_all([union] + filled)
        if not regions:
            # NOTE: Kerb strips close the ground-level surface: pockets between kerb and lanes (junction corners, gore
            # areas) are holes as well, but not the cracks between consecutive kerbs. The kerbs themselves are no road.
            # A tiny closing seals the hairline leaks between kerb ends and lanes.
            closed = shapely.union_all([union, kerb_area], grid_size=0.001)
            closed = closed.buffer(_SEAL, join_style="mitre").buffer(-_SEAL, join_style="mitre")
            pockets = [shapely.Polygon(r) for p in shapely.get_parts(closed) for r in p.interiors]
            filled_area = shapely.union_all(filled)
            filled += [
                h
                for h in pockets
                if h.area > _MIN_POCKET_AREA
                and h.distance(kerb_area) < 0.01
                and not h.representative_point().within(filled_area)
                and not keep_pocket(h)
            ]
            # NOTE: Snapping closes the hairline slits between kerb pockets and lanes. Kerbs with road on both sides
            # (along filled gore areas) are road as well.
            region = closed_union([union] + filled)
            half = KERB_MAX_WIDTH / 2
            closing = region.buffer(half, join_style="mitre").buffer(-half, join_style="mitre")
            sandwiched = [p for p in shapely.get_parts(closing.intersection(kerb_area)) if p.geom_type == "Polygon"]
            filled += [p for p in sandwiched if shapely.maximum_inscribed_circle(p, 0.01).length > 0.1]
            region = closed_union([union] + filled)
        fillers += [shapely.Polygon(lift(shapely.get_coordinates(f.exterior))) for f in filled]
        regions.append(region)
        lifts.append(lift)

    # Other lane types become outline-only surfaces (not raised and not part of a dropped surface piece). Sidewalks
    # and curb strips (the kerb between road edge and sidewalk) are walkways; adjacent ones of a section side merge.
    dropped = shapely.union_all(dropped)
    polygon_of = lambda lane: shapely.Polygon(np.vstack([lane["inner"], lane["outer"][::-1]])[:, :2]).buffer(0)
    is_walkway = lambda k: (lanes[k]["type"] == "sidewalk" or k in curb_lanes) and len(lanes[k]["inner"]) > 1
    walkway_keys = sorted(
        filter(is_walkway, lanes),
        key=lambda k: (lanes[k]["road"], lanes[k]["section"], np.sign(lanes[k]["id"]), abs(lanes[k]["id"])),
    )
    walkway_runs = []
    for key in walkway_keys:
        prev = lanes[walkway_runs[-1][-1]] if walkway_runs else None
        lane = lanes[key]
        prev_next = (prev["road"], prev["section"], prev["id"] + np.sign(lane["id"])) if prev else None
        if prev and prev_next == (lane["road"], lane["section"], lane["id"]):
            walkway_runs[-1].append(key)
        else:
            walkway_runs.append([key])
    road_area = shapely.union_all(regions)
    walkways, clipped = [], []
    for run in walkway_runs:
        first, last = lanes[run[0]], lanes[run[-1]]
        outline = np.vstack([first["inner"], last["outer"][::-1], first["inner"][:1]])
        polygon = shapely.make_valid(shapely.Polygon(outline[:, :2]))
        # NOTE: Junction sidewalk lanes crossing the carriageway are pedestrian crossings, not walkways.
        on_road = polygon.intersection(road_area).area
        if on_road > 0.5 * polygon.area:
            continue
        if on_road > _MIN_WALKWAY_ON_ROAD:
            clipped.append((run[0], outline, polygon))
            continue
        walkways.append(polygon)
        yield Walkway(object_id=run[0], outline=Polyline3D.from_array(outline))
    # NOTE: Kerbs running on into the carriageway (junction roads) are walkways off the road and other walkways only.
    covered = shapely.union_all([road_area] + walkways)
    for key, outline, polygon in clipped:
        lift = cKDTree(outline[:, :2])
        pieces = [p for p in shapely.get_parts(polygon.difference(covered)) if p.geom_type == "Polygon"]
        pieces = [
            p
            for p in pieces
            if p.area >= _MIN_WALKWAY_ON_ROAD and shapely.maximum_inscribed_circle(p, 0.01).length > 0.1
        ]
        for n, piece in enumerate(pieces):
            xy = shapely.get_coordinates(piece.exterior)
            piece_outline = np.column_stack([xy, outline[lift.query(xy)[1], 2]])
            yield Walkway(object_id=f"{key}_{n}", outline=Polyline3D.from_array(piece_outline))
    for key, lane in lanes.items():
        surface = (
            lane["type"] in _SURFACE_TYPES
            and key not in raised
            and not polygon_of(lane).representative_point().within(dropped)
        )
        if surface and not is_walkway(key) and len(lane["inner"]) > 1:
            outline = np.vstack([lane["inner"], lane["outer"][::-1], lane["inner"][:1]])
            yield _SURFACE_TYPES[lane["type"]](object_id=key, outline=Polyline3D.from_array(outline))

    # Road edges: boundaries of the filled surface of each layer, without the seams where layers join.
    runs = []
    for n, region in enumerate(regions):
        other = regions[1 - n]
        region = shapely.orient_polygons(region, exterior_cw=True)
        for ring in [r for p in shapely.get_parts(region) for r in [p.exterior, *p.interiors]]:
            coords = lifts[n](shapely.get_coordinates(ring))
            mid = 0.5 * (coords[:-1] + coords[1:])
            seam = np.zeros(len(mid), dtype=bool)
            if not other.is_empty:
                near = shapely.distance(other, shapely.points(mid[:, :2])) < 0.01
                seam = near & (np.abs(lifts[1 - n](mid[:, :2])[:, 2] - mid[:, 2]) < _LAYER_DZ)
            cuts = np.flatnonzero(np.diff(np.concatenate([[1], seam, [1]])))
            runs += [coords[a : b + 1] for a, b in zip(cuts[::2], cuts[1::2]) if b > a]
    edges = shapely.line_merge(shapely.MultiLineString(runs), directed=True)
    edges = [e for e in shapely.get_parts(edges) if e.length > 0.05]
    for n, edge in enumerate(split_line_geometry_by_max_length(edges, _MAX_ROAD_EDGE_LENGTH)):
        yield RoadEdge(
            object_id=n, road_edge_type=RoadEdgeType.ROAD_EDGE_BOUNDARY, polyline=Polyline3D.from_linestring(edge)
        )

    # NOTE: Generic drivables cover the whole road surface: every road-level lane (driving,
    # biking and parking lanes included, overlapping their Lane / Carpark) and the filled holes.
    for key in surface_keys:
        if _SURFACE_TYPES.get(lanes[key]["type"]) is not GenericDrivable:
            outline = np.vstack([lanes[key]["inner"], lanes[key]["outer"][::-1], lanes[key]["inner"][:1]])
            yield GenericDrivable(object_id=key, outline=Polyline3D.from_array(outline))
    for n, filler in enumerate(fillers):
        yield GenericDrivable(
            object_id=f"filler_{n}",
            outline=Polyline3D.from_array(shapely.get_coordinates(filler.exterior, include_z=True)),
        )

    # Intersections: outline of the union of all road-level lanes of the junction's connecting roads.
    by_junction = {}
    for key, polygon in zip(surface_keys, surfaces):
        if lanes[key]["junction"] != -1:
            by_junction.setdefault(lanes[key]["junction"], []).append(polygon)
    for junction_id, polygons in by_junction.items():
        group_ids = sorted({g for g in groups if roads[int(g.split("_")[0])]["junction"] == junction_id})
        if not group_ids:
            continue
        polygon = max(shapely.get_parts(closed_union(polygons)), key=lambda p: p.area)
        yield Intersection(
            object_id=junction_id,
            intersection_type=IntersectionType.DEFAULT,
            lane_group_ids=group_ids,
            outline=Polyline3D.from_array(lifter(polygons)(shapely.get_coordinates(polygon.exterior))),
        )
