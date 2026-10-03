import math
import string

import numpy as np

from models.photo_geometry import PhotoCorner, PhotoGeometry, PhotoWall
from modules.image_processing.plane_finder import Plane, PlaneFinder, SurfaceClassifier


MIN_WALL_HEIGHT_M = 1.2
MIN_WALL_TOP_M = 1.6
MERGE_ANGLE_DEG = 6
MERGE_DISTANCE_M = 0.15
CORNER_MIN_ANGLE_DEG = 60
CORNER_REACH_M = 0.6
BORDER_FRACTION = 0.03


class FloorFrame:

    def __init__(self, up: np.ndarray, floor_offset):
        self.up = up / np.linalg.norm(up)
        self.floor_offset = floor_offset

        forward = np.array([0.0, 0.0, 1.0])
        forward = forward - (forward @ self.up) * self.up
        self.forward = forward / np.linalg.norm(forward)
        self.right = np.cross(self.forward, self.up)

    def to_2d(self, points: np.ndarray) -> np.ndarray:
        return np.stack([points @ self.right, points @ self.forward], axis=-1)

    def height(self, points: np.ndarray) -> np.ndarray:
        if self.floor_offset is None:
            return points @ self.up
        return points @ self.up + self.floor_offset

    def to_3d(self, point_2d, height: float) -> np.ndarray:
        floor_offset = self.floor_offset if self.floor_offset is not None else 0.0
        return point_2d[0] * self.right + point_2d[1] * self.forward + (height - floor_offset) * self.up


class WallLine:

    def __init__(self, plane, frame: FloorFrame):
        self.plane = plane

        normal_2d = np.array([plane.normal @ frame.right, plane.normal @ frame.forward])
        size = np.linalg.norm(normal_2d)
        self.normal = normal_2d / size
        self.offset = plane.offset / size
        self.direction = np.array([-self.normal[1], self.normal[0]])

        positions = frame.to_2d(plane.points) @ self.direction
        self.start_s = float(np.percentile(positions, 1))
        self.end_s = float(np.percentile(positions, 99))
        self.start_type = "occluded"
        self.end_type = "occluded"

        heights = frame.height(plane.points)
        self.bottom = float(np.percentile(heights, 2))
        self.top = float(np.percentile(heights, 98))

        self.letter = ""

    def point_at(self, position: float) -> np.ndarray:
        return -self.offset * self.normal + position * self.direction

    def position_of(self, point_2d: np.ndarray) -> float:
        return float(point_2d @ self.direction)


class PhotoAnalysis:

    def __init__(self, geometry: PhotoGeometry, frame: FloorFrame, wall_lines: list, intrinsics: np.ndarray):
        self.geometry = geometry
        self.frame = frame
        self.wall_lines = wall_lines
        self.intrinsics = intrinsics

    def wall_line(self, letter: str):
        for wall_line in self.wall_lines:
            if wall_line.letter == letter:
                return wall_line
        return None


class PhotoMeasurer:

    def __init__(self):
        self.plane_finder = PlaneFinder()
        self.classifier = SurfaceClassifier()

    def measure(self, photo_id: str, image_path: str, depth: dict, ignore_mask=None) -> PhotoAnalysis:
        points = depth["points"]
        mask = depth["mask"]
        intrinsics = depth["intrinsics"]
        image_height, image_width = mask.shape

        if ignore_mask is not None:
            mask = mask & ~ignore_mask

        planes = self.plane_finder.find_planes(points, mask)
        surfaces = self.classifier.classify(planes)

        floor_offset = surfaces.floor.offset if surfaces.floor is not None else None
        frame = FloorFrame(surfaces.up, floor_offset)

        notes = []
        if surfaces.floor is None:
            notes.append("floor not found, heights are not reliable")

        wall_planes = self.merge_similar_planes(surfaces.walls)
        wall_lines = self.keep_real_walls(wall_planes, frame, surfaces.floor is not None)
        wall_lines = self.sort_and_name(wall_lines, image_width)

        corners = self.find_corners(wall_lines)
        self.mark_truncated_ends(wall_lines, frame, image_width)

        geometry = PhotoGeometry(
            photo_id=photo_id,
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            fov_x_deg=math.degrees(2 * math.atan(0.5 / float(intrinsics[0, 0]))),
            floor_found=surfaces.floor is not None,
            camera_height_m=floor_offset,
            ceiling_height_m=self.ceiling_height(surfaces),
            walls=self.to_photo_walls(wall_lines, image_width),
            corners=corners,
            notes=notes,
        )

        return PhotoAnalysis(geometry, frame, wall_lines, intrinsics)

    def ceiling_height(self, surfaces):
        if surfaces.floor is None or surfaces.ceiling is None:
            return None
        return float(surfaces.floor.offset + surfaces.ceiling.offset)

    def merge_similar_planes(self, planes: list) -> list:
        merged = []
        angle_limit = np.cos(np.radians(MERGE_ANGLE_DEG))

        for plane in sorted(planes, key=Plane.point_count, reverse=True):
            target = None

            for kept in merged:
                if plane.normal @ kept.normal > angle_limit and abs(plane.offset - kept.offset) < MERGE_DISTANCE_M:
                    target = kept
                    break

            if target is None:
                merged.append(plane)
                continue

            target.points = np.concatenate([target.points, plane.points])
            target.rows = np.concatenate([target.rows, plane.rows])
            target.cols = np.concatenate([target.cols, plane.cols])

        return merged

    def keep_real_walls(self, planes: list, frame: FloorFrame, floor_found: bool) -> list:
        wall_lines = []

        for plane in planes:
            wall_line = WallLine(plane, frame)
            visible_height = wall_line.top - wall_line.bottom

            if visible_height < MIN_WALL_HEIGHT_M:
                continue

            if floor_found and wall_line.top < MIN_WALL_TOP_M:
                continue

            wall_lines.append(wall_line)

        return wall_lines

    def sort_and_name(self, wall_lines: list, image_width: int) -> list:
        centres = []
        for wall_line in wall_lines:
            centres.append(float(np.median(wall_line.plane.cols)) / image_width)

        order = np.argsort(centres)
        sorted_lines = []

        for position, index in enumerate(order):
            wall_line = wall_lines[index]
            wall_line.letter = string.ascii_uppercase[position]
            sorted_lines.append(wall_line)

        return sorted_lines

    def intersection(self, line_a: WallLine, line_b: WallLine):
        matrix = np.array([line_a.normal, line_b.normal])
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None
        return np.linalg.solve(matrix, -np.array([line_a.offset, line_b.offset]))

    def corner_candidates(self, wall_lines: list) -> list:
        candidates = []
        min_cross = np.sin(np.radians(CORNER_MIN_ANGLE_DEG))

        for line_a in wall_lines:
            for line_b in wall_lines:
                if line_a is line_b:
                    continue

                cross = line_a.direction[0] * line_b.direction[1] - line_a.direction[1] * line_b.direction[0]
                if abs(cross) < min_cross:
                    continue

                point = self.intersection(line_a, line_b)
                if point is None:
                    continue

                gap_a = abs(line_a.position_of(point) - line_a.end_s)
                gap_b = abs(line_b.position_of(point) - line_b.start_s)

                if gap_a < CORNER_REACH_M and gap_b < CORNER_REACH_M:
                    candidates.append({"gap": gap_a + gap_b, "line_a": line_a, "line_b": line_b, "point": point, "cross": cross})

        return candidates

    def gap_of(self, candidate: dict) -> float:
        return candidate["gap"]

    def find_corners(self, wall_lines: list) -> list:
        corners = []
        used_ends = set()
        used_starts = set()

        for candidate in sorted(self.corner_candidates(wall_lines), key=self.gap_of):
            line_a = candidate["line_a"]
            line_b = candidate["line_b"]
            point = candidate["point"]

            if line_a.letter in used_ends or line_b.letter in used_starts:
                continue

            used_ends.add(line_a.letter)
            used_starts.add(line_b.letter)

            line_a.end_s = line_a.position_of(point)
            line_a.end_type = "corner"
            line_b.start_s = line_b.position_of(point)
            line_b.start_type = "corner"

            turn_angle = math.degrees(math.acos(float(np.clip(line_a.direction @ line_b.direction, -1, 1))))
            corner_type = "normal" if candidate["cross"] < 0 else "protruding"

            corners.append(PhotoCorner(
                number=len(corners) + 1,
                wall_before=line_a.letter,
                wall_after=line_b.letter,
                point=(round(float(point[0]), 3), round(float(point[1]), 3)),
                angle_deg=round(180 - turn_angle, 1),
                corner_type=corner_type,
            ))

        return corners

    def mark_truncated_ends(self, wall_lines: list, frame: FloorFrame, image_width: int):
        for wall_line in wall_lines:
            positions = frame.to_2d(wall_line.plane.points) @ wall_line.direction
            cols = wall_line.plane.cols / image_width
            span = max(wall_line.end_s - wall_line.start_s, 0.05)

            if wall_line.start_type != "corner":
                near_start = positions < wall_line.start_s + 0.1 * span
                if self.touches_border(cols[near_start]):
                    wall_line.start_type = "truncated"

            if wall_line.end_type != "corner":
                near_end = positions > wall_line.end_s - 0.1 * span
                if self.touches_border(cols[near_end]):
                    wall_line.end_type = "truncated"

    def touches_border(self, cols: np.ndarray) -> bool:
        if len(cols) == 0:
            return False
        at_border = (cols < BORDER_FRACTION) | (cols > 1 - BORDER_FRACTION)
        return bool(at_border.mean() > 0.2)

    def to_photo_walls(self, wall_lines: list, image_width: int) -> list:
        photo_walls = []

        for wall_line in wall_lines:
            start = wall_line.point_at(wall_line.start_s)
            end = wall_line.point_at(wall_line.end_s)

            photo_walls.append(PhotoWall(
                letter=wall_line.letter,
                start=(round(float(start[0]), 3), round(float(start[1]), 3)),
                end=(round(float(end[0]), 3), round(float(end[1]), 3)),
                visible_length_m=round(wall_line.end_s - wall_line.start_s, 3),
                start_type=wall_line.start_type,
                end_type=wall_line.end_type,
                distance_from_camera_m=round(float(wall_line.offset), 3),
                visible_height_m=round(wall_line.top - wall_line.bottom, 3),
                image_center_x=round(float(np.median(wall_line.plane.cols)) / image_width, 3),
                point_count=wall_line.plane.point_count(),
            ))

        return photo_walls
