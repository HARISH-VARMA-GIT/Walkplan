import json
import logging
import os

from modules.floor_plan_generator.main import FloorPlanRenderer
from modules.lidar_processing.capture_reader import StrayCapture
from modules.lidar_processing.debug_drawer import DebugDrawer
from modules.lidar_processing.opening_detector import OpeningDetector
from modules.lidar_processing.opening_placer import OpeningPlacer
from modules.lidar_processing.plan_builder import LidarPlanBuilder
from modules.lidar_processing.plan_frame import PlanFrameFinder
from modules.lidar_processing.plan_grid import PlanGrid
from modules.lidar_processing.point_cloud_builder import PointCloudBuilder, PointCloudLoader
from modules.lidar_processing.room_segmenter import RoomSegmenter, WallLineFinder


CAPTURE_FILES = ["odometry.csv", "camera_matrix.csv"]

logger = logging.getLogger(__name__)


class LidarOutputFolder:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.point_cloud_path = os.path.join(folder_path, "point_cloud.npz")
        self.ply_path = os.path.join(folder_path, "points.ply")
        self.plan_frame_path = os.path.join(folder_path, "plan_frame.json")
        self.detections_path = os.path.join(folder_path, "opening_detections.json")
        self.openings_path = os.path.join(folder_path, "openings.json")
        self.openings_folder = os.path.join(folder_path, "openings")
        self.debug_path = os.path.join(folder_path, "plan_debug.png")
        self.floor_plan_path = os.path.join(folder_path, "floor_plan.json")
        self.svg_path = os.path.join(folder_path, "floor_plan.svg")

        os.makedirs(folder_path, exist_ok=True)

    def save_json(self, path: str, data):
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2)

    def load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)


class LidarProcessingService:

    def __init__(self):
        self.cloud_builder = PointCloudBuilder()
        self.cloud_loader = PointCloudLoader()
        self.frame_finder = PlanFrameFinder()
        self.line_finder = WallLineFinder()
        self.segmenter = RoomSegmenter()
        self.opening_detector = OpeningDetector()
        self.renderer = FloorPlanRenderer()

    def is_capture(self, folder_path: str) -> bool:
        for file_name in CAPTURE_FILES:
            if not os.path.exists(os.path.join(folder_path, file_name)):
                return False
        return os.path.isdir(os.path.join(folder_path, "depth"))

    def find_capture(self, lidar_path: str) -> str:
        if self.is_capture(lidar_path):
            return lidar_path

        captures = []
        for name in sorted(os.listdir(lidar_path)):
            child = os.path.join(lidar_path, name)
            if os.path.isdir(child) and self.is_capture(child):
                captures.append(child)

        if not captures:
            raise FileNotFoundError(f"No Stray Scanner capture (odometry.csv, camera_matrix.csv, depth/) found in {lidar_path}")
        if len(captures) > 1:
            raise ValueError(f"Found {len(captures)} captures in {lidar_path}; put one capture per folder: {captures}")

        return captures[0]

    def room_name_for(self, lidar_path: str) -> str:
        return os.path.basename(os.path.normpath(lidar_path))

    def load_cloud(self, capture: StrayCapture, output: LidarOutputFolder, redo: bool):
        if os.path.exists(output.point_cloud_path) and not redo:
            logger.info("Using cached point cloud %s", output.point_cloud_path)
            return self.cloud_loader.load(output.point_cloud_path)

        logger.info("Fusing LiDAR depth frames into a point cloud")
        cloud = self.cloud_builder.build(capture)
        cloud.save(output.point_cloud_path)
        cloud.save_ply(output.ply_path)
        return cloud

    def detect_openings(self, capture: StrayCapture, output: LidarOutputFolder, redo: bool) -> list:
        if os.path.exists(output.detections_path) and not redo:
            logger.info("Using cached door and window detections")
            return output.load_json(output.detections_path)

        detections = self.opening_detector.detect(capture, output.openings_folder)
        output.save_json(output.detections_path, detections)
        return detections

    def room_summary(self, room) -> dict:
        return {"id": f"R{room.index}", "area_m2": round(room.area, 2), "walls": len(room.edges)}

    def build_floor_plan(self, lidar_path: str, output_folder: str, room_type: str = "other", redo_lidar: bool = False,
                         find_openings: bool = True) -> dict:
        capture_path = self.find_capture(lidar_path)
        capture = StrayCapture(capture_path)
        output = LidarOutputFolder(output_folder)
        logger.info("LiDAR capture %s: %d frames, depth %dx%d", capture_path, len(capture.poses), capture.depth_width, capture.depth_height)

        cloud = self.load_cloud(capture, output, redo_lidar)
        frame = self.frame_finder.find(cloud)
        output.save_json(output.plan_frame_path, frame.to_dict())
        logger.info("Floor found, ceiling %s m, walls turned %.1f deg", frame.ceiling_height_m, frame.wall_angle_deg)

        grid = PlanGrid(cloud, frame)
        x_lines = self.line_finder.find(grid, 0)
        y_lines = self.line_finder.find(grid, 1)
        segmented = self.segmenter.segment(grid, x_lines, y_lines)
        rooms = segmented["rooms"]
        logger.info("Found %d wall lines and %d rooms", len(x_lines) + len(y_lines), len(rooms))

        placer = OpeningPlacer(capture, frame)
        camera_openings = []
        if find_openings:
            camera_openings = placer.place(self.detect_openings(capture, output, redo_lidar), rooms)
        openings = placer.merge(camera_openings, placer.gap_openings(grid, segmented["door_gaps"], rooms))
        output.save_json(output.openings_path, openings)

        builder = LidarPlanBuilder(cloud, frame, grid)
        plan = builder.build(rooms, openings, capture.name, room_type, self.segmenter.notes)
        output.save_json(output.floor_plan_path, plan.model_dump(mode="json"))
        self.renderer.save_svg(plan, output.svg_path)
        DebugDrawer(cloud, frame).save(rooms, openings, output.debug_path)

        return {
            "capture_path": capture_path,
            "floor_plan": plan.model_dump(mode="json"),
            "floor_plan_path": output.floor_plan_path,
            "svg_path": output.svg_path,
            "rooms": [self.room_summary(room) for room in rooms],
        }
