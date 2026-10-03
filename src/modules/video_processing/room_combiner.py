import math

import numpy as np

from models.floor_plan import Adjacency, Room
from modules.image_processing.photo_placer import PhotoPlacer
from utils.geometry import FloorPlanGeometry


DOOR_MATCH_M = 0.4
DOOR_SNAP_M = 1.2
ROOM_GAP_M = 1.0


class PlanFrame:

    def __init__(self, placements, placer: PhotoPlacer):
        self.up = np.array(placements.up, dtype=float)
        self.axis_x, self.axis_y = placer.world_axes(self.up)
        self.scale = placements.scale
        self.wall_angle = math.radians(placements.wall_angle_deg)

    def to_plan(self, world_point: np.ndarray) -> np.ndarray:
        point = np.array([world_point @ self.axis_x, world_point @ self.axis_y]) * self.scale
        cos_value = math.cos(-self.wall_angle)
        sin_value = math.sin(-self.wall_angle)
        return np.array([cos_value * point[0] - sin_value * point[1], sin_value * point[0] + cos_value * point[1]])


class PlanMove:

    def __init__(self, quarter_turns: int, shift: np.ndarray):
        self.quarter_turns = quarter_turns % 4
        self.shift = shift

    def rotate(self, point) -> np.ndarray:
        x, y = point
        for turn in range(self.quarter_turns):
            x, y = -y, x
        return np.array([x, y])

    def apply(self, point) -> tuple:
        moved = self.rotate(point) + self.shift
        return round(float(moved[0]), 3), round(float(moved[1]), 3)


class RoomCombiner:

    def __init__(self):
        self.placer = PhotoPlacer()
        self.geometry = FloorPlanGeometry()
        self.notes = []

    def fit_move(self, plan_points: list, target_points: list) -> PlanMove:
        source = np.array(plan_points)
        target = np.array(target_points)
        source_centre = source.mean(axis=0)
        target_centre = target.mean(axis=0)

        a = source - source_centre
        b = target - target_centre
        angle = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])), float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
        quarter_turns = int(round(angle / (math.pi / 2)))

        move = PlanMove(quarter_turns, np.zeros(2))
        rotated = np.array([move.rotate(point) for point in source])
        move.shift = target_centre - rotated.mean(axis=0)

        residuals = np.linalg.norm(rotated + move.shift - target, axis=1)
        self.notes.append(f"room placed with {quarter_turns % 4 * 90} deg turn, camera fit error {float(np.median(residuals)):.2f} m")
        return move

    def move_room(self, room: Room, move: PlanMove) -> Room:
        moved = room.model_copy(deep=True)

        for wall in moved.walls:
            wall.start = move.apply(wall.start)
            wall.end = move.apply(wall.end)

        for damage in moved.damage:
            if damage.point is not None:
                damage.point = move.apply(damage.point)

        return moved

    def room_bounds(self, room: Room) -> tuple:
        xs = []
        ys = []
        for wall in room.walls:
            xs.extend([wall.start[0], wall.end[0]])
            ys.extend([wall.start[1], wall.end[1]])
        return min(xs), max(xs), min(ys), max(ys)

    def place_beside(self, room: Room, placed_rooms: list) -> PlanMove:
        right_edge = max(self.room_bounds(other)[1] for other in placed_rooms)
        top_edge = max(self.room_bounds(other)[3] for other in placed_rooms)
        min_x, max_x, min_y, max_y = self.room_bounds(room)
        return PlanMove(0, np.array([right_edge + ROOM_GAP_M - min_x, top_edge - max_y]))

    def door_centre(self, room: Room, opening) -> np.ndarray:
        for wall in room.walls:
            if wall.id != opening.wall_id:
                continue

            start = np.array(wall.start)
            direction = np.array(wall.end) - start
            direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
            return start + direction * (opening.offset_m + opening.width.value / 2)

        return None

    def door_wall(self, room: Room, opening):
        for wall in room.walls:
            if wall.id == opening.wall_id:
                return wall
        return None

    def outward_normal(self, room: Room, wall) -> np.ndarray:
        direction = np.array(wall.end) - np.array(wall.start)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)

        if self.geometry.is_counter_clockwise(room):
            return np.array([direction[1], -direction[0]])
        return np.array([-direction[1], direction[0]])

    def snap_to_door(self, room: Room, placed_rooms: list) -> Room:
        best = None

        for opening in room.openings:
            if opening.type != "door":
                continue
            wall = self.door_wall(room, opening)
            centre = self.door_centre(room, opening)

            for other in placed_rooms:
                for other_opening in other.openings:
                    if other_opening.type != "door":
                        continue
                    other_wall = self.door_wall(other, other_opening)
                    facing = self.outward_normal(room, wall) @ self.outward_normal(other, other_wall)
                    if facing > -0.9:
                        continue

                    target = self.door_centre(other, other_opening) + self.outward_normal(other, other_wall) * other_wall.thickness_m
                    shift = target - centre
                    distance = float(np.linalg.norm(shift))

                    if distance < DOOR_SNAP_M and (best is None or distance < best[0]):
                        best = (distance, shift, opening.id, other_opening.id)

        if best is None:
            self.notes.append(f"{room.id}: no shared door found to snap to, position from camera poses only")
            return room

        distance, shift, door_id, other_door_id = best
        self.notes.append(f"{room.id}: moved {distance:.2f} m to line up {door_id} with {other_door_id}")
        return self.move_room(room, PlanMove(0, shift))

    def link_doors(self, rooms: list) -> list:
        adjacency = []

        for index, room in enumerate(rooms):
            for other in rooms[index + 1:]:
                best = None
                for opening in room.openings:
                    if opening.type != "door":
                        continue
                    centre = self.door_centre(room, opening)

                    for other_opening in other.openings:
                        if other_opening.type != "door":
                            continue
                        distance = float(np.linalg.norm(centre - self.door_centre(other, other_opening)))

                        if distance < DOOR_MATCH_M and (best is None or distance < best[0]):
                            best = (distance, opening, other_opening)

                if best is not None:
                    distance, opening, other_opening = best
                    opening.leads_to = other.id
                    other_opening.leads_to = room.id
                    adjacency.append(Adjacency(room_a=room.id, room_b=other.id, via_opening=opening.id))
                    self.notes.append(f"{room.id} and {other.id} connected through {opening.id} ({distance:.2f} m apart)")

        return adjacency

    def rename(self, room: Room, room_id: str, name: str, label: str) -> Room:
        room.id = room_id
        room.name = name
        room.label = label

        for opening in room.openings:
            opening.id = f"{room_id}-{opening.id}"

        return room

    def combine(self, room_results: list, global_transforms: list) -> dict:
        self.notes = []
        reference = PlanFrame(room_results[0]["placements"], self.placer)
        placed = []

        for result, transform in zip(room_results, global_transforms):
            room = Room.model_validate(result["room"])
            room = self.rename(room, f"R{result['index']}", result["name"], result["room_type"])

            if not placed:
                placed.append(room)
                continue

            if transform is None:
                move = self.place_beside(room, placed)
                self.notes.append(f"{room.id} could not be lined up with the other rooms, drawn beside them")
                placed.append(self.move_room(room, move))
                continue

            plan_points = []
            target_points = []
            for placement, pose in zip(result["placements"].placements, result["poses"]):
                world_pose = transform.apply_to_pose(pose["camera_to_world"])
                plan_points.append(placement.position)
                target_points.append(reference.to_plan(world_pose[:3, 3]))

            moved = self.move_room(room, self.fit_move(plan_points, target_points))
            placed.append(self.snap_to_door(moved, placed))

        adjacency = self.link_doors(placed)
        return {"rooms": placed, "adjacency": adjacency, "notes": self.notes}
