import logging
import math
import os

import cv2
import numpy as np

from modules.image_processing.opening_finder import OpeningFinder
from modules.lidar_processing.capture_reader import StrayCapture


KEYFRAME_SECONDS = 0.6
MAX_KEYFRAMES = 300
MAX_PITCH_DEG = 50
MAX_PIXEL_SAMPLES = 1500
EDGE_MARGIN_PX = 3
DRAWN_FRAMES = 40
DETECT_MAX_SIDE = 1024

logger = logging.getLogger(__name__)


class KeyframePicker:

    def rotation_change(self, capture: StrayCapture, frame_index: int) -> float:
        before = capture.poses.get(frame_index - 2)
        after = capture.poses.get(frame_index + 2)
        if before is None or after is None:
            return math.inf

        relative = before.rotation.T @ after.rotation
        cosine = (np.trace(relative) - 1) / 2
        return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))

    def pitch_deg(self, capture: StrayCapture, frame_index: int) -> float:
        forward = capture.poses[frame_index].direction_to_world(np.array([0.0, 0.0, 1.0]))
        return float(np.degrees(np.arcsin(np.clip(forward[1], -1, 1))))

    def pick(self, capture: StrayCapture) -> list:
        frame_indexes = sorted(capture.poses)
        start_time = capture.poses[frame_indexes[0]].time_seconds
        duration = capture.poses[frame_indexes[-1]].time_seconds - start_time

        window = max(KEYFRAME_SECONDS, duration / MAX_KEYFRAMES)
        windows = {}

        for frame_index in frame_indexes:
            if abs(self.pitch_deg(capture, frame_index)) > MAX_PITCH_DEG:
                continue

            slot = int((capture.poses[frame_index].time_seconds - start_time) / window)
            change = self.rotation_change(capture, frame_index)
            if slot not in windows or change < windows[slot][0]:
                windows[slot] = (change, frame_index)

        picked = []
        for change, frame_index in windows.values():
            if math.isfinite(change):
                picked.append(frame_index)

        return sorted(picked)


class FrameReader:

    def read(self, video_path: str, frame_indexes: list):
        wanted = set(frame_indexes)
        last = max(frame_indexes)
        video = cv2.VideoCapture(video_path)

        frame_index = 0
        while frame_index <= last:
            if not video.grab():
                break

            if frame_index in wanted:
                ok, image_bgr = video.retrieve()
                if ok:
                    yield frame_index, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            frame_index += 1

        video.release()


class OpeningDetector:

    def __init__(self):
        self.opening_finder = OpeningFinder()
        self.keyframe_picker = KeyframePicker()
        self.frame_reader = FrameReader()

    def upright_turn(self, capture: StrayCapture, frame_index: int):
        up = capture.poses[frame_index].direction_to_camera(np.array([0.0, 1.0, 0.0]))

        if abs(up[0]) > abs(up[1]):
            if up[0] < 0:
                return cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE
            return cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE

        if up[1] > 0:
            return cv2.ROTATE_180, cv2.ROTATE_180

        return None, None

    def turn(self, image: np.ndarray, code):
        if code is None:
            return image
        return cv2.rotate(image, code)

    def median_depth(self, capture: StrayCapture, frame_index: int, columns: np.ndarray, rows: np.ndarray):
        file_name = capture.depth_files.get(frame_index)
        if file_name is None:
            return None

        depth = capture.read_depth(file_name)
        depth_columns = np.clip((columns * capture.depth_width / capture.image_width).astype(int), 0, capture.depth_width - 1)
        depth_rows = np.clip((rows * capture.depth_height / capture.image_height).astype(int), 0, capture.depth_height - 1)
        values = depth[depth_rows, depth_columns]
        values = values[values > 0]

        if len(values) < 10:
            return None
        return float(np.median(values))

    def is_cut(self, box: list, label: str, width: int) -> bool:
        cut_side = box[0] < EDGE_MARGIN_PX or box[2] > width - EDGE_MARGIN_PX
        cut_top = box[1] < EDGE_MARGIN_PX and label == "door"
        return bool(cut_side or cut_top)

    def describe(self, capture: StrayCapture, frame_index: int, detection: dict, mask: np.ndarray, scale: float, upright_width: int) -> dict:
        rows, columns = np.nonzero(mask)
        if len(rows) > MAX_PIXEL_SAMPLES:
            pick = np.linspace(0, len(rows) - 1, MAX_PIXEL_SAMPLES).astype(int)
            rows = rows[pick]
            columns = columns[pick]

        rows = np.round((rows + 0.5) / scale - 0.5)
        columns = np.round((columns + 0.5) / scale - 0.5)

        return {
            "frame_index": frame_index,
            "label": detection["label"],
            "score": round(detection["score"], 3),
            "cut_at_edge": self.is_cut(detection["box"], detection["label"], upright_width),
            "median_depth": self.median_depth(capture, frame_index, columns, rows),
            "columns": columns.astype(int).tolist(),
            "rows": rows.astype(int).tolist(),
        }

    def draw(self, image_rgb: np.ndarray, detections: list, path: str):
        drawn = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        for detection in detections:
            color = (40, 160, 40) if detection["label"] == "door" else (213, 123, 58)
            box = [int(value) for value in detection["box"]]
            cv2.rectangle(drawn, (box[0], box[1]), (box[2], box[3]), color, 3)
            cv2.putText(drawn, f"{detection['label']} {detection['score']:.2f}", (box[0] + 6, box[1] + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite(path, drawn)

    def detect(self, capture: StrayCapture, drawing_folder: str) -> list:
        frame_indexes = self.keyframe_picker.pick(capture)
        logger.info("Looking for doors and windows in %d keyframes", len(frame_indexes))

        os.makedirs(drawing_folder, exist_ok=True)
        found = []
        drawn = 0

        for count, (frame_index, image_rgb) in enumerate(self.frame_reader.read(capture.video_path, frame_indexes)):
            turn_code, back_code = self.upright_turn(capture, frame_index)
            scale = min(1.0, DETECT_MAX_SIDE / max(image_rgb.shape[:2]))
            small = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            upright = self.turn(small, turn_code)

            detections = self.opening_finder.find_openings(upright)
            for detection in detections:
                mask = self.turn(detection["mask"].astype(np.uint8), back_code) > 0
                found.append(self.describe(capture, frame_index, detection, mask, scale, upright.shape[1]))

            if detections and drawn < DRAWN_FRAMES:
                self.draw(upright, detections, os.path.join(drawing_folder, f"frame_{frame_index:06d}.jpg"))
                drawn += 1

            if (count + 1) % 50 == 0:
                logger.info("Checked %d of %d keyframes (%d doors/windows seen)", count + 1, len(frame_indexes), len(found))

        self.opening_finder.unload_models()
        return found
