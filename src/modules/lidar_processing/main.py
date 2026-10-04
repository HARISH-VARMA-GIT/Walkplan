import json
import logging
import os

from modules.floor_plan_generator.main import FloorPlanRenderer
from modules.image_processing.opening_finder import OPENING_WORDS
from modules.lidar_processing.capture_reader import StrayCapture
from modules.lidar_processing.debug_drawer import DebugDrawer
from modules.lidar_processing.mirror_cleaner import MirrorCleaner
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
        self.mirrors_path = os.path.join(folder_path, "mirrors.json")
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
        if not os.path.isdir(lidar_path):
            raise FileNotFoundError(f"LiDAR folder not found: {lidar_path}")

        if self.is_capture(lidar_path):
            return lidar_path

        captures = self.captures_below(lidar_path)
        if not captures:
            for name in sorted(os.listdir(lidar_path)):
                child = os.path.join(lidar_path, name)
                if os.path.isdir(child):
                    captures.extend(self.captures_below(child))

        if not captures:
            raise FileNotFoundError(f"No Stray Scanner capture (odometry.csv, camera_matrix.csv, depth/) found in {lidar_path}")
        if len(captures) > 1:
            raise ValueError(f"Found {len(captures)} captures in {lidar_path}; put one capture per folder: {captures}")

        return captures[0]

    def captures_below(self, folder_path: str) -> list:
        captures = []

        for name in sorted(os.listdir(folder_path)):
            child = os.path.join(folder_path, name)
            if os.path.isdir(child) and self.is_capture(child):
                captures.append(child)

        return captures

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

    def cached_detections(self, output: LidarOutputFolder):
        if not os.path.exists(output.detections_path):
            return None

        cached = output.load_json(output.detections_path)
        if not isinstance(cached, dict) or cached.get("words") != OPENING_WORDS:
            logger.info("Cached detections were made with other words, looking again")
            return None

        return cached["detections"]

    def detect_openings(self, capture: StrayCapture, output: LidarOutputFolder, redo: bool) -> list:
        cached = None if redo else self.cached_detections(output)
        if cached is not None:
            logger.info("Using cached door, window and mirror detections")
            return cached

        detections = self.opening_detector.detect(capture, output.openings_folder)
        output.save_json(output.detections_path, {"words": OPENING_WORDS, "detections": detections})
        return detections

    def find_rooms(self, cloud, frame) -> dict:
        grid = PlanGrid(cloud, frame)
        x_lines = self.line_finder.find(grid, 0)
        y_lines = self.line_finder.find(grid, 1)
        segmented = self.segmenter.segment(grid, x_lines, y_lines)
        logger.info("Found %d wall lines and %d rooms", len(x_lines) + len(y_lines), len(segmented["rooms"]))

        return {"grid": grid, "rooms": segmented["rooms"], "door_gaps": segmented["door_gaps"], "notes": list(self.segmenter.notes)}

    def mirror_notes(self, mirrors: list) -> list:
        notes = []

        for mirror in mirrors:
            notes.append(f"R{mirror['room_index']}: mirror on {mirror['wall_id']} at {mirror['start']:.2f}–{mirror['end']:.2f} m "
                         f"({mirror['views']} views), not counted as a window or door")

        return notes

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

        found = self.find_rooms(cloud, frame)
        placer = OpeningPlacer(capture, frame)
        detections = []
        if find_openings:
            detections = self.detect_openings(capture, output, redo_lidar)
        placed = placer.place(detections, found["rooms"])

        if placed["mirrors"]:
            cleaned = MirrorCleaner(frame).clean(cloud, placed["mirrors"], found["rooms"])
            if cleaned["removed"] > 0:
                cloud = cleaned["cloud"]
                found = self.find_rooms(cloud, frame)
                placed = placer.place(detections, found["rooms"])

        rooms = found["rooms"]
        gap_openings = placer.gap_openings(found["grid"], found["door_gaps"], rooms)
        openings = placer.merge(placed["openings"], gap_openings, placed["mirrors"])
        output.save_json(output.openings_path, openings)
        output.save_json(output.mirrors_path, placed["mirrors"])

        notes = found["notes"] + self.mirror_notes(placed["mirrors"])
        builder = LidarPlanBuilder(cloud, frame, found["grid"])
        plan = builder.build(rooms, openings, capture.name, room_type, notes)
        output.save_json(output.floor_plan_path, plan.model_dump(mode="json"))
        self.renderer.save_svg(plan, output.svg_path)
        DebugDrawer(cloud, frame).save(rooms, openings, placed["mirrors"], output.debug_path)

        return {
            "capture_path": capture_path,
            "floor_plan": plan.model_dump(mode="json"),
            "floor_plan_path": output.floor_plan_path,
            "svg_path": output.svg_path,
            "rooms": [self.room_summary(room) for room in rooms],
        }
