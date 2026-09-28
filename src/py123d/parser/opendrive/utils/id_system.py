def lane_key(road_id: int, section_idx: int, lane_id: int) -> str:
    """Builds the py123d id of an OpenDRIVE lane.

    :param road_id: OpenDRIVE road id.
    :param section_idx: Lane section index within the road.
    :param lane_id: OpenDRIVE lane id (positive: left, negative: right).
    :return: Lane id, e.g. "12_0_right_-1".
    """
    return f"{road_id}_{section_idx}_{'left' if lane_id > 0 else 'right'}_{lane_id}"
