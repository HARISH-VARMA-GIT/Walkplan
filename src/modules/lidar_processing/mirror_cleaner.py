import logging

import numpy as np
from scipy.spatial import cKDTree

from modules.lidar_processing.plan_frame import PlanFrame
from modules.lidar_processing.point_cloud_builder import PointCloud
from modules.lidar_processing.room_segmenter import RoomFinder


MIN_BEHIND_M = 0.1
MAX_BEHIND_M = 4.0
MATCH_DISTANCE_M = 0.04
SIDE_SPREAD = 1.0

logger = logging.getLogger(__name__)


class MirrorCleaner:

    def __init__(self, frame: PlanFrame):
        self.frame = frame

    def reflected_points(self, plan_points: np.ndarray, heights: np.ndarray, tree: cKDTree, mirror: dict, edge) -> np.ndarray:
        normal = np.array(edge.normal, dtype=float)
        signed = (plan_points - edge.start) @ normal
        behind = -signed
        along = (plan_points - edge.start) @ edge.direction

        side_room = behind * SIDE_SPREAD
        candidates = (behind > MIN_BEHIND_M) & (behind < MAX_BEHIND_M)
        candidates &= (along > mirror["start"] - side_room) & (along < mirror["end"] + side_room)

        found = np.zeros(len(plan_points), dtype=bool)
        indexes = np.flatnonzero(candidates)
        if len(indexes) == 0:
            return found

        mirrored = plan_points[indexes] + 2 * behind[indexes, None] * normal
        distances, nearest = tree.query(np.c_[mirrored, heights[indexes]], distance_upper_bound=MATCH_DISTANCE_M)
        found[indexes[distances < MATCH_DISTANCE_M]] = True
        return found

    def clean(self, cloud: PointCloud, mirrors: list, rooms: list) -> dict:
        plan_points = self.frame.to_plan(cloud.points)
        heights = self.frame.height(cloud.points)
        tree = cKDTree(np.c_[plan_points, heights])
        finder = RoomFinder(rooms)

        remove = np.zeros(len(plan_points), dtype=bool)
        for mirror in mirrors:
            room = finder.room_by_index(mirror["room_index"])
            edge = finder.edge_by_id(room, mirror["wall_id"])
            found = self.reflected_points(plan_points, heights, tree, mirror, edge)
            mirror["reflected_points"] = int(found.sum())
            remove |= found

        logger.info("Removed %d reflected points behind %d mirrors", int(remove.sum()), len(mirrors))
        cleaned = PointCloud(cloud.points[~remove], cloud.normals[~remove], cloud.counts[~remove], cloud.camera_path)
        return {"cloud": cleaned, "removed": int(remove.sum())}
