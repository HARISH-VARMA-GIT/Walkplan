import math

import numpy as np

from models.floor_plan import Damage, Measurement, Opening, Room, Wall
from models.image_models import RoomLayout
from models.photo_geometry import RoomPhotoGeometry
from modules.image_processing.wall_position_solver import WallPositionSolver


RELATIVE_ERROR = 0.05
MIN_WALL_LENGTH_M = 0.05
MIN_VISIBLE_LENGTH_M = 0.1
MIN_VOTE_LENGTH_M = 0.3
HEADING_AGREEMENT_DEG = 20
MATCH_ANGLE_DEG = 25
SOURCE = "moge2_photo"
DAMAGE_AREA_RELATIVE_ERROR = 0.5
DAMAGE_CONFIDENCE = {"measured": 0.7, "estimated": 0.5, "assumed": 0.3}


class RoomAssembler:

    def __init__(self):
        self.notes = []

    def build_wall_matches(self, layout: RoomLayout) -> dict:
        matches = {}

        for match in layout.wall_matches:
            if match.room_wall_id.lower() != "none":
                matches[(match.photo_id, match.wall_letter)] = match.room_wall_id

        return matches

    def direction_angle(self, direction) -> float:
        return math.degrees(math.atan2(direction[1], direction[0]))

    def angle_gap(self, angle_a: float, angle_b: float) -> float:
        return abs((angle_a - angle_b + 180) % 360 - 180)

    def photo_wall_angle(self, wall) -> float:
        return self.direction_angle(np.array(wall.end) - np.array(wall.start))

    def best_heading(self, photo, llm_matches: dict, wall_indexes: dict, directions: list):
        votes = []
        for wall in photo.walls:
            room_wall_id = llm_matches.get((photo.photo_id, wall.letter))
            if room_wall_id in wall_indexes and wall.visible_length_m >= MIN_VOTE_LENGTH_M:
                room_angle = self.direction_angle(directions[wall_indexes[room_wall_id]])
                votes.append((room_angle - self.photo_wall_angle(wall), wall.visible_length_m))

        best_heading = None
        best_score = 0.0

        for heading, weight in votes:
            score = 0.0
            for other_heading, other_weight in votes:
                if self.angle_gap(heading, other_heading) < HEADING_AGREEMENT_DEG:
                    score += other_weight
            if score > best_score:
                best_heading = heading
                best_score = score

        return best_heading

    def repair_matches(self, photo_set: RoomPhotoGeometry, layout: RoomLayout, wall_indexes: dict, directions: list) -> dict:
        llm_matches = self.build_wall_matches(layout)
        repaired = {}
        changed = 0

        for photo in photo_set.photos:
            heading = self.best_heading(photo, llm_matches, wall_indexes, directions)
            if heading is None:
                continue

            for wall in photo.walls:
                if wall.visible_length_m < MIN_VISIBLE_LENGTH_M:
                    continue

                room_angle = self.photo_wall_angle(wall) + heading
                candidates = []
                for wall_info in layout.walls:
                    if self.angle_gap(room_angle, self.direction_angle(directions[wall_indexes[wall_info.id]])) < MATCH_ANGLE_DEG:
                        candidates.append(wall_info.id)

                llm_choice = llm_matches.get((photo.photo_id, wall.letter))

                if len(candidates) == 1:
                    choice = candidates[0]
                elif llm_choice in candidates:
                    choice = llm_choice
                else:
                    choice = None

                if choice is not None:
                    repaired[(photo.photo_id, wall.letter)] = choice
                if choice != llm_choice:
                    changed += 1

        self.notes.append(f"geometry changed {changed} of the vision model's wall matches")
        return repaired

    def wall_directions(self, layout: RoomLayout) -> list:
        directions = []
        direction = np.array([1.0, 0.0])
        right_turns = 0
        left_turns = 0

        for wall_info in layout.walls:
            directions.append(direction.copy())

            if wall_info.corner_after == "normal":
                direction = np.array([direction[1], -direction[0]])
                right_turns += 1
            else:
                direction = np.array([-direction[1], direction[0]])
                left_turns += 1

        if right_turns - left_turns != 4:
            raise ValueError(f"Layout does not close a loop ({right_turns} normal, {left_turns} protruding corners). "
                             f"Fix layout.json and run again.")

        return directions

    def photo_wall_normal(self, wall) -> np.ndarray:
        direction = np.array(wall.end) - np.array(wall.start)
        direction = direction / np.linalg.norm(direction)
        return np.array([direction[1], -direction[0]])

    def add_photos_to_solver(self, solver: WallPositionSolver, photo_set: RoomPhotoGeometry, wall_indexes: dict, matches: dict, placements=None):
        placement_by_id = {}
        if placements is not None:
            for placement in placements.placements:
                placement_by_id[placement.photo_id] = placement

        for photo in photo_set.photos:
            seen_walls = []

            for wall in photo.walls:
                room_wall_id = matches.get((photo.photo_id, wall.letter))
                if room_wall_id not in wall_indexes or wall.visible_length_m < MIN_VISIBLE_LENGTH_M:
                    continue

                seen_walls.append({
                    "wall_index": wall_indexes[room_wall_id],
                    "normal": self.photo_wall_normal(wall),
                    "distance": wall.distance_from_camera_m,
                    "end_points": [np.array(wall.start), np.array(wall.end)],
                })

            placement = placement_by_id.get(photo.photo_id)
            if placement is None:
                solver.add_photo(photo.photo_id, seen_walls)
            else:
                solver.add_photo(photo.photo_id, seen_walls, math.radians(placement.heading_deg), placement.position)

    def build_walls(self, layout: RoomLayout, solver: WallPositionSolver) -> list:
        observed = solver.observed_walls()
        walls = []

        for index, wall_info in enumerate(layout.walls):
            start, end = solver.wall_ends(index)
            length = float(np.linalg.norm(end - start))

            if length < MIN_WALL_LENGTH_M:
                self.notes.append(f"{wall_info.id}: came out {length:.2f} m long, check the layout")

            before, after = solver.neighbours(index)
            if index in observed and before in observed and after in observed:
                method = "measured"
            else:
                method = "estimated"
                self.notes.append(f"{wall_info.id}: length estimated, the wall or a neighbour was not seen")

            sigma = max(solver.wall_length_sigma(index), RELATIVE_ERROR * length, 0.02)

            walls.append(Wall(
                id=wall_info.id,
                start=(round(float(start[0]), 3), round(float(start[1]), 3)),
                end=(round(float(end[0]), 3), round(float(end[1]), 3)),
                length=Measurement(
                    value=round(length, 3),
                    low=round(length - sigma, 3),
                    high=round(length + sigma, 3),
                    method=method,
                    source=SOURCE,
                ),
            ))

        return walls

    def interval(self, values: list, minimum_sigma: float, method: str = "measured") -> Measurement:
        value = float(np.median(values))
        sigma = max((max(values) - min(values)) / 2, minimum_sigma)
        return Measurement(value=round(value, 3), low=round(value - sigma, 3), high=round(value + sigma, 3),
                           method=method, source=SOURCE)

    def group_openings(self, photo_set: RoomPhotoGeometry, layout: RoomLayout, wall_indexes: dict, matches: dict) -> dict:
        photo_openings = {}
        for photo in photo_set.photos:
            for opening in photo.openings:
                photo_openings[opening.id] = (photo, opening)

        groups = {}
        for match in layout.opening_matches:
            if match.type == "none" or match.room_opening_id.lower() == "none":
                continue
            if match.photo_opening_id not in photo_openings or match.room_wall_id not in wall_indexes:
                continue

            if match.room_opening_id not in groups:
                groups[match.room_opening_id] = {"type": match.type, "wall_id": match.room_wall_id, "items": []}
            groups[match.room_opening_id]["items"].append(photo_openings[match.photo_opening_id])

        for group in groups.values():
            group["wall_id"] = self.most_common_wall(group, matches)

        return groups

    def most_common_wall(self, group: dict, matches: dict) -> str:
        counts = {}
        for photo, opening in group["items"]:
            wall_id = matches.get((photo.photo_id, opening.wall_letter))
            if wall_id is not None:
                counts[wall_id] = counts.get(wall_id, 0) + 1

        if not counts:
            return group["wall_id"]

        return max(counts, key=counts.get)

    def opening_point_in_photo(self, photo, opening):
        for wall in photo.walls:
            if wall.letter != opening.wall_letter:
                continue

            start = np.array(wall.start)
            direction = np.array(wall.end) - start
            direction = direction / np.linalg.norm(direction)
            return start + opening.start_along_wall_m * direction

        return None

    def build_openings(self, photo_set: RoomPhotoGeometry, layout: RoomLayout, wall_indexes: dict,
                       matches: dict, solver: WallPositionSolver, walls: list) -> list:
        groups = self.group_openings(photo_set, layout, wall_indexes, matches)
        openings = []

        for room_opening_id, group in groups.items():
            opening = self.build_one_opening(room_opening_id, group, wall_indexes, matches, solver, walls)
            if opening is not None:
                openings.append(opening)

        return openings

    def build_one_opening(self, room_opening_id: str, group: dict, wall_indexes: dict, matches: dict,
                          solver: WallPositionSolver, walls: list):
        wall_id = group["wall_id"]
        wall_index = wall_indexes[wall_id]
        wall_length = walls[wall_index].length.value

        measured = []
        for photo, opening in group["items"]:
            if opening.width_m is not None:
                measured.append((photo, opening))

        if not measured:
            self.notes.append(f"{room_opening_id}: no usable measurement, left out")
            return None

        trusted = []
        for photo, opening in measured:
            if opening.looks_valid:
                trusted.append((photo, opening))
        if not trusted:
            trusted = measured
            self.notes.append(f"{room_opening_id}: size outside normal range in every photo, check it")

        widths = [opening.width_m for photo, opening in trusted]
        width = self.interval(widths, 0.05)

        offsets = []
        for photo, opening in trusted:
            if matches.get((photo.photo_id, opening.wall_letter)) != wall_id:
                continue

            point = self.opening_point_in_photo(photo, opening)
            if point is None:
                continue

            offset = solver.position_along_wall(photo.photo_id, wall_index, point)
            if offset is not None:
                offsets.append(offset)

        if offsets:
            offset = float(np.median(offsets))
        else:
            offset = (wall_length - width.value) / 2
            self.notes.append(f"{room_opening_id}: position along {wall_id} not measured, placed in the middle")

        offset = float(np.clip(offset, 0, max(wall_length - width.value, 0)))

        heights = [opening.height_m for photo, opening in trusted if opening.height_m is not None]
        sills = [opening.sill_height_m for photo, opening in trusted if opening.sill_height_m is not None]

        return Opening(
            id=room_opening_id,
            type=group["type"],
            wall_id=wall_id,
            offset_m=round(offset, 3),
            width=width,
            height=self.interval(heights, 0.05) if heights else None,
            sill_height=self.interval(sills, 0.05) if sills and group["type"] == "window" else None,
            swing="left" if group["type"] == "door" else "none",
        )

    def build_ceiling(self, photo_set: RoomPhotoGeometry):
        heights = []
        for photo in photo_set.photos:
            if photo.ceiling_height_m is not None and 2.0 <= photo.ceiling_height_m <= 6.0:
                heights.append(photo.ceiling_height_m)

        if not heights:
            self.notes.append("ceiling not found in any photo")
            return None

        return self.interval(heights, 0.05)

    def room_point(self, photo_id: str, point, solver: WallPositionSolver, placements):
        if photo_id in solver.poses and solver.poses[photo_id].position is not None:
            return solver.poses[photo_id].to_room(point)

        if placements is None:
            return None

        for placement in placements.placements:
            if placement.photo_id == photo_id:
                heading = math.radians(placement.heading_deg)
                x, y = point
                rotated = np.array([math.cos(heading) * x - math.sin(heading) * y, math.sin(heading) * x + math.cos(heading) * y])
                return np.array(placement.position) + rotated

        return None

    def closest_wall(self, room_point, walls: list) -> int:
        best_index = 0
        best_distance = None

        for index, wall in enumerate(walls):
            start = np.array(wall.start)
            direction = np.array(wall.end) - start
            length = max(float(np.linalg.norm(direction)), 1e-6)
            along = float(np.clip((room_point - start) @ direction / length, 0, length))
            distance = float(np.linalg.norm(start + direction / length * along - room_point))

            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance

        return best_index

    def collect_photo_damages(self, photo_set: RoomPhotoGeometry) -> dict:
        found = {}
        for photo in photo_set.photos:
            for damage in photo.damages:
                found.setdefault(damage.damage_id, []).append((photo, damage))
        return found

    def build_damages(self, photo_set: RoomPhotoGeometry, damage_details: dict, matches: dict,
                      solver: WallPositionSolver, walls: list, placements) -> list:
        photo_damages = self.collect_photo_damages(photo_set)
        damage_ids = list(damage_details.keys())
        for damage_id in photo_damages:
            if damage_id not in damage_ids:
                damage_ids.append(damage_id)

        damages = []
        for damage_id in damage_ids:
            details = damage_details.get(damage_id, {})
            damages.append(self.build_one_damage(damage_id, details, photo_damages.get(damage_id, []), matches, solver, walls, placements))

        return damages

    def build_one_damage(self, damage_id: str, details: dict, items: list, matches: dict,
                         solver: WallPositionSolver, walls: list, placements) -> Damage:
        damage = Damage(
            id=damage_id,
            surface_type="wall",
            damage_class=details.get("damage_class", "other"),
            confidence=0.2,
            severity=details.get("severity"),
            description=details.get("description"),
            quote=details.get("quote"),
            notes=details.get("location") or None,
            video_time_s=details.get("start_seconds"),
            source_frames=details.get("source_frames", []),
        )

        if details.get("surface_type") in ["floor", "ceiling"]:
            damage.surface_type = details["surface_type"]

        by_method = {"measured": [], "estimated": [], "assumed": []}
        for photo, item in items:
            if item.point is not None:
                by_method[item.method].append((photo, item))

        method = None
        for name in ["measured", "estimated", "assumed"]:
            if method is None and by_method[name]:
                method = name

        if method is None:
            self.notes.append(f"{damage_id}: damage not found in any frame, position unknown")
            return damage

        usable = by_method[method]
        damage.location_method = method
        damage.confidence = DAMAGE_CONFIDENCE[method]

        surface_counts = {}
        for photo, item in usable:
            surface_counts[item.surface_type] = surface_counts.get(item.surface_type, 0) + 1
        damage.surface_type = max(surface_counts, key=surface_counts.get)

        on_surface = []
        room_points = []
        wall_votes = {}
        for photo, item in usable:
            if item.surface_type != damage.surface_type:
                continue

            room_point = self.room_point(photo.photo_id, item.point, solver, placements)
            if room_point is None:
                continue

            on_surface.append(item)
            room_points.append(room_point)

            wall_id = matches.get((photo.photo_id, item.wall_letter))
            if wall_id is not None:
                wall_votes[wall_id] = wall_votes.get(wall_id, 0) + 1

        if not room_points:
            self.notes.append(f"{damage_id}: frames have no camera position, position unknown")
            return damage

        centre = np.median(np.array(room_points), axis=0)

        if damage.surface_type == "wall":
            centre = self.place_on_wall(damage, centre, wall_votes, walls)

        damage.point = (round(float(centre[0]), 3), round(float(centre[1]), 3))

        heights = [item.center_height_m for item in on_surface if item.center_height_m is not None]
        if heights:
            damage.height_m = round(float(np.median(heights)), 3)

        widths = [item.width_m for item in on_surface if item.width_m is not None]
        if widths:
            damage.width_m = round(float(np.median(widths)), 3)

        areas = [item.area_m2 for item in on_surface if item.area_m2 is not None]
        if areas:
            area = float(np.median(areas))
            damage.extent_m2 = Measurement(
                value=round(area, 3),
                low=round(area * (1 - DAMAGE_AREA_RELATIVE_ERROR), 3),
                high=round(area * (1 + DAMAGE_AREA_RELATIVE_ERROR), 3),
                method=method,
                source=SOURCE,
            )

        if method != "measured":
            self.notes.append(f"{damage_id}: damage outline not detected, position {method} from the frame")

        return damage

    def place_on_wall(self, damage: Damage, centre, wall_votes: dict, walls: list):
        wall_index = None
        if wall_votes:
            best_wall_id = max(wall_votes, key=wall_votes.get)
            for index, wall in enumerate(walls):
                if wall.id == best_wall_id:
                    wall_index = index

        if wall_index is None:
            wall_index = self.closest_wall(centre, walls)

        wall = walls[wall_index]
        start = np.array(wall.start)
        direction = np.array(wall.end) - start
        length = max(float(np.linalg.norm(direction)), 1e-6)
        direction = direction / length

        offset = float(np.clip((centre - start) @ direction, 0, length))
        damage.wall_id = wall.id
        damage.offset_m = round(offset, 3)

        return start + direction * offset

    def assemble(self, photo_set: RoomPhotoGeometry, layout: RoomLayout, placements=None, damage_details: dict = None) -> dict:
        self.notes = []

        if len(layout.walls) < 3:
            raise ValueError("Layout has fewer than 3 walls, cannot build a room")

        wall_indexes = {}
        for index, wall_info in enumerate(layout.walls):
            wall_indexes[wall_info.id] = index

        directions = self.wall_directions(layout)

        if placements is None:
            matches = self.repair_matches(photo_set, layout, wall_indexes, directions)
        else:
            matches = self.build_wall_matches(layout)

        solver = WallPositionSolver(directions)
        self.add_photos_to_solver(solver, photo_set, wall_indexes, matches, placements)

        if placements is not None:
            for index, wall_info in enumerate(layout.walls):
                if wall_info.position_m is not None:
                    solver.set_wall_prior(index, wall_info.position_m)
        solver.solve()

        walls = self.build_walls(layout, solver)
        openings = self.build_openings(photo_set, layout, wall_indexes, matches, solver, walls)

        room = Room(
            id="R1",
            label=layout.room_type,
            name=photo_set.room_name,
            walls=walls,
            openings=openings,
            ceiling_height=self.build_ceiling(photo_set),
            damage=self.build_damages(photo_set, damage_details or {}, matches, solver, walls, placements),
        )

        matches_used = []
        for (photo_id, wall_letter), room_wall_id in matches.items():
            matches_used.append({"photo_id": photo_id, "wall_letter": wall_letter, "room_wall_id": room_wall_id})

        notes = solver.notes + self.notes
        return {"room": room, "notes": notes, "photos_used": len(solver.poses), "matches_used": matches_used}
