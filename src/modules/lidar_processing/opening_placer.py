import numpy as np

from modules.image_processing.opening_finder import VALID_SIZES
from modules.lidar_processing.capture_reader import StrayCapture
from modules.lidar_processing.plan_frame import PlanFrame
from modules.lidar_processing.plan_grid import CELL_SIZE_M, PlanGrid
from modules.lidar_processing.room_segmenter import RoomFinder


MAX_WALL_DISTANCE_M = 0.4
EDGE_OVERHANG_M = 0.2
INNER_SIDE_M = 0.05
SAME_OPENING_M = 0.5
MIN_VIEWS = 2
SINGLE_VIEW_SCORE = 0.5
DOOR_FLOOR_GAP_M = 0.25
MIN_WINDOW_SILL_M = 0.2
GAP_WALL_DISTANCE_M = 0.3
GAP_LOOK_ACROSS_M = 0.5
GAP_MIN_FREE_SHARE = 0.3
GAP_LOW_WALL_SHARE = 0.3
LOW_WALL_TOP_M = 0.6
DEFAULT_DOOR_HEIGHT_M = 2.03
MAX_WINDOW_SILL_M = 1.6
DEFAULT_MAX_TOP_M = 2.9
TOP_ALLOWANCE_M = 0.1
WINDOW_JOIN_GAP_M = 0.15
OUTSIDE_REACH_M = 0.45


class OpeningPlacer:

    def __init__(self, capture: StrayCapture, frame: PlanFrame):
        self.capture = capture
        self.frame = frame
        self.max_top = DEFAULT_MAX_TOP_M if frame.ceiling_height_m is None else frame.ceiling_height_m + TOP_ALLOWANCE_M

    def pixel_rays(self, columns: np.ndarray, rows: np.ndarray) -> np.ndarray:
        matrix = self.capture.camera_matrix
        x = (columns + 0.5 - matrix[0, 2]) / matrix[0, 0]
        y = (rows + 0.5 - matrix[1, 2]) / matrix[1, 1]
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def is_inner_side(self, edge, point: np.ndarray) -> bool:
        return float(np.array(edge.normal) @ (point - edge.start)) > INNER_SIDE_M

    def wall_near_point(self, rooms: list, point: np.ndarray, camera: np.ndarray):
        best = None
        best_distance = MAX_WALL_DISTANCE_M

        for room in rooms:
            for edge in room.edges:
                along = edge.along(point)
                if along < -EDGE_OVERHANG_M or along > edge.length + EDGE_OVERHANG_M or not self.is_inner_side(edge, camera):
                    continue

                distance = abs(float(np.array(edge.normal) @ (point - edge.start)))
                if distance < best_distance:
                    best = (room, edge)
                    best_distance = distance

        return best

    def first_wall_crossed(self, rooms: list, camera: np.ndarray, direction: np.ndarray):
        best = None
        best_step = None

        for room in rooms:
            for edge in room.edges:
                facing = float(np.array(edge.normal) @ direction)
                if facing >= -1e-6 or not self.is_inner_side(edge, camera):
                    continue

                step = float(np.array(edge.normal) @ (edge.start - camera)) / facing
                along = edge.along(camera + step * direction)
                if step <= 0 or along < -EDGE_OVERHANG_M or along > edge.length + EDGE_OVERHANG_M:
                    continue

                if best_step is None or step < best_step:
                    best = (room, edge)
                    best_step = step

        return best

    def place_one(self, detection: dict, rooms: list):
        pose = self.capture.poses[detection["frame_index"]]
        rays = self.pixel_rays(np.array(detection["columns"], dtype=float), np.array(detection["rows"], dtype=float))
        if len(rays) < 20:
            return None

        world_directions = pose.direction_to_world(rays)
        plan_directions = self.frame.direction_to_plan(world_directions)
        vertical = world_directions[:, 1]
        camera = self.frame.to_plan(pose.position)
        camera_height = float(self.frame.height(pose.position))

        centre_ray = rays.mean(axis=0)
        chosen = None
        if detection["median_depth"] is not None:
            centre_point = self.frame.to_plan(pose.to_world(centre_ray * detection["median_depth"]))
            chosen = self.wall_near_point(rooms, centre_point, camera)
        if chosen is None:
            chosen = self.first_wall_crossed(rooms, camera, self.frame.direction_to_plan(pose.direction_to_world(centre_ray)))
        if chosen is None:
            return None

        room, edge = chosen
        normal = np.array(edge.normal, dtype=float)
        facing = plan_directions @ normal
        usable = facing < -1e-6
        if usable.sum() < 20:
            return None

        steps = (normal @ (edge.start - camera)) / facing[usable]
        hits = camera + steps[:, None] * plan_directions[usable]
        along = (hits - edge.start) @ edge.direction
        heights = camera_height + steps * vertical[usable]

        near = float(np.percentile(along, 2))
        far = float(np.percentile(along, 98))
        bottom = float(np.percentile(heights, 2))
        top = float(np.percentile(heights, 98))

        return {
            "room_index": room.index,
            "wall_id": edge.wall_id,
            "type": detection["label"],
            "start": near,
            "end": far,
            "bottom": bottom,
            "top": top,
            "score": detection["score"],
            "frame_index": detection["frame_index"],
            "valid": self.is_valid(detection, far - near, bottom, top),
        }

    def is_valid(self, detection: dict, width: float, bottom: float, top: float) -> bool:
        if detection["cut_at_edge"]:
            return False

        limits = VALID_SIZES[detection["label"]]
        height = top if detection["label"] == "door" else top - bottom

        if not limits["width"][0] <= width <= limits["width"][1]:
            return False
        if not limits["height"][0] <= height <= limits["height"][1]:
            return False

        if top > self.max_top:
            return False
        if detection["label"] == "door":
            return bottom < DOOR_FLOOR_GAP_M
        return MIN_WINDOW_SILL_M < bottom < MAX_WINDOW_SILL_M

    def centre_of(self, item: dict) -> float:
        return (item["start"] + item["end"]) / 2

    def cluster(self, placed: list) -> list:
        by_wall = {}
        for item in placed:
            if item is not None and item["valid"]:
                by_wall.setdefault((item["room_index"], item["wall_id"], item["type"]), []).append(item)

        openings = []
        for (room_index, wall_id, opening_type), items in by_wall.items():
            items.sort(key=self.centre_of)

            group = []
            for item in items + [None]:
                if item is not None and (not group or self.centre_of(item) - self.centre_of(group[-1]) < SAME_OPENING_M):
                    group.append(item)
                    continue

                if len(group) >= MIN_VIEWS or max(member["score"] for member in group) >= SINGLE_VIEW_SCORE:
                    openings.append(self.summarise(room_index, wall_id, opening_type, group))
                group = [item] if item is not None else []

        return openings

    def summarise(self, room_index: int, wall_id: str, opening_type: str, group: list) -> dict:
        starts = [member["start"] for member in group]
        ends = [member["end"] for member in group]
        widths = [member["end"] - member["start"] for member in group]
        tops = [member["top"] for member in group]
        bottoms = [member["bottom"] for member in group]

        return {
            "room_index": room_index,
            "wall_id": wall_id,
            "type": opening_type,
            "start": float(np.median(starts)),
            "end": float(np.median(ends)),
            "width_spread": float(np.std(widths)) if len(widths) > 1 else 0.05,
            "top": float(np.median(tops)),
            "bottom": float(np.median(bottoms)),
            "height_spread": float(np.std(tops)) if len(tops) > 1 else 0.05,
            "views": len(group),
            "frames": [member["frame_index"] for member in group],
            "source": "camera",
        }

    def join_windows(self, openings: list) -> list:
        joined = []

        for item in sorted(openings, key=self.start_of):
            last = joined[-1] if joined else None
            same_wall = last is not None and last["type"] == "window" and item["type"] == "window" and                 last["room_index"] == item["room_index"] and last["wall_id"] == item["wall_id"]

            if same_wall and item["start"] - last["end"] < WINDOW_JOIN_GAP_M:
                last["end"] = max(last["end"], item["end"])
                last["top"] = max(last["top"], item["top"])
                last["bottom"] = min(last["bottom"], item["bottom"])
                last["views"] += item["views"]
                last["frames"] = last["frames"] + item["frames"]
            else:
                joined.append(item)

        return joined

    def start_of(self, item: dict) -> tuple:
        return item["room_index"], item["wall_id"], item["start"]

    def outside_windows_only(self, openings: list, rooms: list) -> list:
        finder = RoomFinder(rooms)
        kept = []

        for item in openings:
            if item["type"] == "window":
                room = finder.room_by_index(item["room_index"])
                edge = finder.edge_by_id(room, item["wall_id"])
                if finder.room_behind(room, edge, self.centre_of(item), OUTSIDE_REACH_M) is not None:
                    continue
            kept.append(item)

        return kept

    def shared_length(self, first: dict, second: dict) -> float:
        return min(first["end"], second["end"]) - max(first["start"], second["start"])

    def windows_not_on_doors(self, openings: list) -> list:
        doors = [item for item in openings if item["type"] == "door"]
        kept = []

        for item in openings:
            covered = False
            if item["type"] == "window":
                for door in doors:
                    same_wall = door["room_index"] == item["room_index"] and door["wall_id"] == item["wall_id"]
                    smaller = min(door["end"] - door["start"], item["end"] - item["start"])
                    if same_wall and self.shared_length(door, item) > 0.5 * smaller:
                        covered = True
            if not covered:
                kept.append(item)

        return kept

    def place(self, detections: list, rooms: list) -> list:
        placed = []
        for detection in detections:
            placed.append(self.place_one(detection, rooms))

        openings = self.outside_windows_only(self.cluster(placed), rooms)
        return self.join_windows(self.windows_not_on_doors(openings))

    def free_across(self, grid: PlanGrid, gap: dict) -> bool:
        axis = gap["axis"]
        shares = []

        for side in (-1, 1):
            low = gap["position"] + side * 0.1
            high = gap["position"] + side * GAP_LOOK_ACROSS_M
            if axis == 0:
                share = grid.free_share(min(low, high), max(low, high), gap["start"], gap["end"])
            else:
                share = grid.free_share(gap["start"], gap["end"], min(low, high), max(low, high))
            shares.append(share)

        return min(shares) >= GAP_MIN_FREE_SHARE

    def has_low_wall(self, grid: PlanGrid, gap: dict) -> bool:
        axis = gap["axis"]
        line_cell = grid.to_cell(gap["position"], axis)
        low_cell = max(line_cell - 1, 0)
        high_cell = min(line_cell + 2, grid.shape[axis])

        if axis == 0:
            lows = grid.wall_low[0][low_cell:high_cell, gap["first_cell"]:gap["last_cell"] + 1].min(axis=0)
        else:
            lows = grid.wall_low[1][gap["first_cell"]:gap["last_cell"] + 1, low_cell:high_cell].min(axis=1)

        return float(np.mean(lows < LOW_WALL_TOP_M)) >= GAP_LOW_WALL_SHARE

    def gap_openings(self, grid: PlanGrid, door_gaps: list, rooms: list) -> list:
        openings = []

        for gap in door_gaps:
            if not self.free_across(grid, gap) or self.has_low_wall(grid, gap):
                continue

            for room in rooms:
                for edge in room.edges:
                    edge_axis = 0 if edge.normal[0] != 0 else 1
                    if edge_axis != gap["axis"] or abs(edge.line_offset() * edge.normal[edge_axis] - gap["position"]) > GAP_WALL_DISTANCE_M:
                        continue

                    if gap["axis"] == 0:
                        first = edge.along(np.array([edge.start[0], gap["start"]]))
                        second = edge.along(np.array([edge.start[0], gap["end"]]))
                    else:
                        first = edge.along(np.array([gap["start"], edge.start[1]]))
                        second = edge.along(np.array([gap["end"], edge.start[1]]))

                    start = max(min(first, second), 0.0)
                    end = min(max(first, second), edge.length)
                    if end - start < 0.5 * (gap["end"] - gap["start"]):
                        continue

                    openings.append({
                        "room_index": room.index,
                        "wall_id": edge.wall_id,
                        "type": "door",
                        "start": float(start),
                        "end": float(end),
                        "width_spread": CELL_SIZE_M,
                        "top": None,
                        "bottom": 0.0,
                        "height_spread": None,
                        "views": 0,
                        "frames": [],
                        "source": "wall_gap",
                    })

        return openings

    def overlaps(self, first: dict, second: dict) -> bool:
        same_wall = first["room_index"] == second["room_index"] and first["wall_id"] == second["wall_id"]
        return same_wall and abs(self.centre_of(first) - self.centre_of(second)) < SAME_OPENING_M

    def merge(self, camera_openings: list, gap_openings: list) -> list:
        merged = list(camera_openings)

        for gap_opening in gap_openings:
            match = None
            for opening in merged:
                if self.overlaps(opening, gap_opening):
                    match = opening

            if match is None:
                merged.append(gap_opening)
                continue

            if match["type"] == "window":
                merged.remove(match)
                merged.append(gap_opening)
                continue

            if match["type"] == "door" and match["source"] == "camera":
                match["start"] = gap_opening["start"]
                match["end"] = gap_opening["end"]
                match["width_spread"] = gap_opening["width_spread"]
                match["source"] = "camera+wall_gap"

        return merged
