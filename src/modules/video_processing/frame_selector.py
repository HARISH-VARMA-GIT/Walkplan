import os
import re
import shutil

import cv2
import numpy as np

from models.video_models import VideoFrame
from utils.ffmpeg_runner import FfmpegRunner


CANDIDATES_PER_SECOND = 2
MAX_SIDE_PIXELS = 1600
SHARPNESS_WIDTH = 640
DAMAGE_FRAMES_EACH = 2
DAMAGE_BEFORE_S = 0.5
DAMAGE_AFTER_S = 2.0
DAMAGE_FRAME_GAP_S = 0.75
MIN_LAYOUT_FRAMES = 8


class FrameSampler:

    def __init__(self, ffmpeg: FfmpegRunner):
        self.ffmpeg = ffmpeg

    def sample(self, video_path: str, candidate_folder: str) -> list:
        if os.path.isdir(candidate_folder):
            shutil.rmtree(candidate_folder)
        os.makedirs(candidate_folder)

        scale = f"scale='min({MAX_SIDE_PIXELS},iw)':-2"
        log = self.ffmpeg.run([
            "-i", video_path,
            "-vf", f"fps={CANDIDATES_PER_SECOND},{scale},showinfo",
            "-fps_mode", "vfr", "-q:v", "2",
            os.path.join(candidate_folder, "c_%05d.jpg"),
        ])

        times = [float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", log)]
        file_names = sorted(os.listdir(candidate_folder))

        candidates = []
        for index, file_name in enumerate(file_names):
            time_seconds = times[index] if index < len(times) else index / CANDIDATES_PER_SECOND
            path = os.path.join(candidate_folder, file_name)
            candidates.append({"path": path, "time_seconds": round(time_seconds, 2), "sharpness": self.sharpness(path)})

        return candidates

    def sharpness(self, image_path: str) -> float:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0

        scale = SHARPNESS_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, (SHARPNESS_WIDTH, int(gray.shape[0] * scale)))
        return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


class FrameSelector:

    def sharpness_of(self, candidate: dict) -> float:
        return candidate["sharpness"]

    def time_of(self, candidate: dict) -> float:
        return candidate["time_seconds"]

    def layout_frames(self, candidates: list, duration: float, count: int) -> list:
        chosen = []
        bin_edges = np.linspace(0, duration, count + 1)

        for index in range(count):
            in_bin = []
            for candidate in candidates:
                if bin_edges[index] <= candidate["time_seconds"] < bin_edges[index + 1]:
                    in_bin.append(candidate)

            if in_bin:
                chosen.append(max(in_bin, key=self.sharpness_of))

        return chosen

    def candidates_between(self, candidates: list, start: float, end: float) -> list:
        found = []
        for candidate in candidates:
            if start <= candidate["time_seconds"] <= end:
                found.append(candidate)
        return found

    def damage_frames(self, candidates: list, damage: dict, duration: float) -> list:
        start = max(damage["start_seconds"] - DAMAGE_BEFORE_S, 0.0)
        in_window = self.candidates_between(candidates, start, min(damage["end_seconds"] + DAMAGE_BEFORE_S, duration))

        if len(in_window) < DAMAGE_FRAMES_EACH:
            in_window = self.candidates_between(candidates, start, min(damage["end_seconds"] + DAMAGE_AFTER_S, duration))

        chosen = []
        for candidate in sorted(in_window, key=self.sharpness_of, reverse=True):
            too_close = False
            for other in chosen:
                if abs(other["time_seconds"] - candidate["time_seconds"]) < DAMAGE_FRAME_GAP_S:
                    too_close = True

            if not too_close:
                chosen.append(candidate)
            if len(chosen) == DAMAGE_FRAMES_EACH:
                break

        return chosen

    def select(self, candidates: list, damages: list, duration: float, max_frames: int) -> list:
        damage_by_path = {}
        for damage in damages:
            for candidate in self.damage_frames(candidates, damage, duration):
                damage_by_path.setdefault(candidate["path"], []).append(damage["id"])

        layout_count = max(MIN_LAYOUT_FRAMES, max_frames - len(damage_by_path))
        layout_paths = set()
        while True:
            layout_paths = set()
            for candidate in self.layout_frames(candidates, duration, layout_count):
                layout_paths.add(candidate["path"])

            total = len(layout_paths | set(damage_by_path.keys()))
            if total >= max_frames or layout_count >= len(candidates):
                break
            layout_count += max_frames - total

        selected = []
        for candidate in sorted(candidates, key=self.time_of):
            is_layout = candidate["path"] in layout_paths
            damage_ids = damage_by_path.get(candidate["path"], [])

            if not is_layout and not damage_ids:
                continue

            if is_layout and damage_ids:
                purpose = "both"
            elif is_layout:
                purpose = "layout"
            else:
                purpose = "damage"

            selected.append({"candidate": candidate, "purpose": purpose, "damage_ids": damage_ids})

        return selected

    def save(self, selected: list, frame_folder: str) -> list:
        if os.path.isdir(frame_folder):
            shutil.rmtree(frame_folder)
        os.makedirs(frame_folder)

        frames = []
        for index, item in enumerate(selected):
            candidate = item["candidate"]
            file_name = f"frame_{index + 1:03d}_{candidate['time_seconds']:06.1f}s.jpg"
            shutil.copyfile(candidate["path"], os.path.join(frame_folder, file_name))

            frames.append(VideoFrame(
                photo_id=f"P{index + 1:02d}",
                file_name=file_name,
                time_seconds=candidate["time_seconds"],
                sharpness=candidate["sharpness"],
                purpose=item["purpose"],
                damage_ids=item["damage_ids"],
            ))

        return frames
