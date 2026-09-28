def lane_key(road_id, section_idx, lane_id):
    return f"{road_id}_{section_idx}_{'left' if lane_id > 0 else 'right'}_{lane_id}"
