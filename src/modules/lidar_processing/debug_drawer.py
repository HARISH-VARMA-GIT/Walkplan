import cv2
import numpy as np

from modules.lidar_processing.plan_frame import PlanFrame
from modules.lidar_processing.point_cloud_builder import PointCloud


PX_PER_M = 80
MARGIN_M = 0.5
CUT_LOW_M = 1.3
CUT_HIGH_M = 1.9
ROOM_COLORS = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148), (75, 86, 140), (194, 119, 227), (34, 189, 188)]
DOOR_COLOR = (40, 160, 40)
WINDOW_COLOR = (213, 123, 58)
MIRROR_COLOR = (200, 40, 200)


class DebugDrawer:

    def __init__(self, cloud: PointCloud, frame: PlanFrame):
        self.plan_points = frame.to_plan(cloud.points)
        self.heights = frame.height(cloud.points)
        self.camera_path = frame.to_plan(cloud.camera_path)
        self.low = None
        self.size = None

    def pixels(self, points: np.ndarray) -> np.ndarray:
        columns = (points[:, 0] - self.low[0]) * PX_PER_M
        rows = self.size[1] - 1 - (points[:, 1] - self.low[1]) * PX_PER_M
        return np.stack([columns, rows], axis=1).astype(int)

    def draw_points(self, image: np.ndarray, points: np.ndarray, color: tuple):
        pixels = self.pixels(points)
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < self.size[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.size[1])
        image[pixels[inside, 1], pixels[inside, 0]] = color

    def save(self, rooms: list, openings: list, mirrors: list, path: str):
        corners = np.array([edge.start for room in rooms for edge in room.edges])
        self.low = corners.min(axis=0) - MARGIN_M
        high = corners.max(axis=0) + MARGIN_M
        self.size = ((high - self.low) * PX_PER_M).astype(int) + 1

        image = np.full((self.size[1], self.size[0], 3), 255, np.uint8)
        self.draw_points(image, self.plan_points[np.abs(self.heights) < 0.05], (205, 235, 205))
        self.draw_points(image, self.plan_points[(self.heights > CUT_LOW_M) & (self.heights < CUT_HIGH_M)], (90, 90, 90))
        cv2.polylines(image, [self.pixels(self.camera_path)], False, (230, 160, 60), 1)

        edges_by_room = {}
        for room in rooms:
            color = ROOM_COLORS[(room.index - 1) % len(ROOM_COLORS)]
            outline = self.pixels(np.array([edge.start for edge in room.edges]))
            cv2.polylines(image, [outline], True, color, 3)

            centre = outline.mean(axis=0).astype(int)
            cv2.putText(image, f"R{room.index} {room.area:.1f} m2", (int(centre[0]) - 40, int(centre[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            for edge in room.edges:
                edges_by_room[(room.index, edge.wall_id)] = edge

        for item in openings + mirrors:
            edge = edges_by_room[(item["room_index"], item["wall_id"])]
            start = edge.start + edge.direction * item["start"]
            end = edge.start + edge.direction * item["end"]
            ends = self.pixels(np.array([start, end]))
            cv2.line(image, tuple(int(value) for value in ends[0]), tuple(int(value) for value in ends[1]), self.color_for(item["type"]), 7)

        cv2.imwrite(path, image)

    def color_for(self, item_type: str) -> tuple:
        if item_type == "door":
            return DOOR_COLOR
        if item_type == "window":
            return WINDOW_COLOR
        return MIRROR_COLOR
