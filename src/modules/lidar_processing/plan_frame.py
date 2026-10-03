import math

import numpy as np

from modules.lidar_processing.point_cloud_builder import PointCloud


HORIZONTAL_NORMAL = 0.9
VERTICAL_NORMAL = 0.3
HEIGHT_BIN_M = 0.02
MIN_CAMERA_ABOVE_FLOOR_M = 0.4
MAX_CAMERA_ABOVE_FLOOR_M = 2.2
MIN_CEILING_HEIGHT_M = 2.0
MAX_CEILING_HEIGHT_M = 4.5
MIN_CEILING_SHARE = 0.15
MIN_CEILING_TO_FLOOR = 0.05
ANGLE_BIN_DEG = 1.0
ANGLE_REFINE_DEG = 3.0


class PlanFrame:

    def __init__(self, floor_y: float, wall_angle_deg: float, ceiling_height_m, floor_spread_m: float):
        self.floor_y = floor_y
        self.wall_angle_deg = wall_angle_deg
        self.ceiling_height_m = ceiling_height_m
        self.floor_spread_m = floor_spread_m

        angle = math.radians(wall_angle_deg)
        self.rotation = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])

    def to_plan(self, world_points: np.ndarray) -> np.ndarray:
        flat = np.stack([world_points[..., 0], -world_points[..., 2]], axis=-1)
        return flat @ self.rotation.T

    def direction_to_plan(self, world_directions: np.ndarray) -> np.ndarray:
        return self.to_plan(world_directions)

    def height(self, world_points: np.ndarray) -> np.ndarray:
        return world_points[..., 1] - self.floor_y

    def to_world(self, plan_points: np.ndarray, heights: np.ndarray) -> np.ndarray:
        flat = plan_points @ self.rotation
        return np.stack([flat[..., 0], heights + self.floor_y, -flat[..., 1]], axis=-1)

    def to_dict(self) -> dict:
        return {
            "floor_y": self.floor_y,
            "wall_angle_deg": self.wall_angle_deg,
            "ceiling_height_m": self.ceiling_height_m,
            "floor_spread_m": self.floor_spread_m,
        }


class PlanFrameFinder:

    def height_peaks(self, heights: np.ndarray, low: float, high: float) -> list:
        if len(heights) == 0 or high <= low:
            return []

        counts, edges = np.histogram(heights, bins=np.arange(low, high + HEIGHT_BIN_M, HEIGHT_BIN_M))
        smooth = np.convolve(counts, [1, 2, 3, 2, 1], mode="same") / 9.0

        peaks = []
        for index in range(1, len(smooth) - 1):
            if smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1] and smooth[index] > 0:
                centre = (edges[index] + edges[index + 1]) / 2
                near = heights[np.abs(heights - centre) < 0.04]
                peaks.append({"height": float(np.median(near)), "count": int(len(near)), "spread": float(np.std(near))})

        return peaks

    def count_of(self, peak: dict) -> int:
        return peak["count"]

    def find_floor(self, cloud: PointCloud) -> dict:
        camera_height = float(np.median(cloud.camera_path[:, 1]))
        floor_points = cloud.points[cloud.normals[:, 1] > HORIZONTAL_NORMAL, 1]

        peaks = self.height_peaks(floor_points, camera_height - MAX_CAMERA_ABOVE_FLOOR_M, camera_height - MIN_CAMERA_ABOVE_FLOOR_M)
        if not peaks:
            raise ValueError("No floor found in the LiDAR scan")

        return max(peaks, key=self.count_of)

    def find_ceiling(self, cloud: PointCloud, floor: dict):
        floor_y = floor["height"]
        ceiling_points = cloud.points[cloud.normals[:, 1] < -HORIZONTAL_NORMAL, 1]

        peaks = self.height_peaks(ceiling_points, floor_y + MIN_CEILING_HEIGHT_M, floor_y + MAX_CEILING_HEIGHT_M)
        if not peaks:
            return None

        best = max(peaks, key=self.count_of)
        if best["count"] < MIN_CEILING_SHARE * len(ceiling_points) or best["count"] < MIN_CEILING_TO_FLOOR * floor["count"]:
            return None

        return best["height"] - floor_y

    def find_wall_angle(self, cloud: PointCloud) -> float:
        wall = np.abs(cloud.normals[:, 1]) < VERTICAL_NORMAL
        normals = cloud.normals[wall]
        angles = np.degrees(np.arctan2(-normals[:, 2], normals[:, 0])) % 90.0

        counts, edges = np.histogram(angles, bins=int(90 / ANGLE_BIN_DEG), range=(0, 90))
        padded = np.concatenate([counts[-3:], counts, counts[:3]])
        smooth = np.convolve(padded, np.ones(5), mode="same")[3:-3]
        peak = (edges[np.argmax(smooth)] + edges[np.argmax(smooth) + 1]) / 2

        gaps = (angles - peak + 45) % 90 - 45
        close = np.abs(gaps) < ANGLE_REFINE_DEG
        return float((peak + np.mean(gaps[close])) % 90)

    def find(self, cloud: PointCloud) -> PlanFrame:
        floor = self.find_floor(cloud)
        ceiling_height = self.find_ceiling(cloud, floor)
        wall_angle = self.find_wall_angle(cloud)
        return PlanFrame(floor["height"], wall_angle, ceiling_height, floor["spread"])
