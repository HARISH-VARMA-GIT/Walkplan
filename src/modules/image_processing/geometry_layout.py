import bisect
import math

import numpy as np

from models.image_models import OpeningMatch, RoomLayout, RoomWallInfo, WallMatch
from models.photo_geometry import RoomPhotoGeometry, RoomPlacements


MIN_WALL_LENGTH_M = 0.3
SNAP_ANGLE_DEG = 20
CLUSTER_GAP_M = 0.25
GRID_MERGE_M = 0.15
SAMPLE_STEP_M = 0.15
RAY_STEP_M = 0.1
MIN_CELL_HITS = 2
RAY_STOP_SHORT_M = 0.3
MIN_HIT_SHARE = 0.25
MATCH_DISTANCE_M = 0.35
MATCH_OVERLAP_M = 0.3
SAME_OPENING_M = 0.5
MIN_END_LINE_LENGTH_M = 1.0
END_LINE_MIN_GAP_M = 0.35
MIN_EDGE_LENGTH_M = 0.2

AXES = [(1, 0), (-1, 0), (0, 1), (0, -1)]
AXIS_NAMES = {(1, 0): "east", (-1, 0): "west", (0, 1): "north", (0, -1): "south"}


class WorldWall:

    def __init__(self, photo_id: str, letter: str, normal: tuple, offset: float, start: np.ndarray, end: np.ndarray, camera: np.ndarray,
                 start_type: str = "occluded", end_type: str = "occluded"):
        self.photo_id = photo_id
        self.start_type = start_type
        self.end_type = end_type
        self.letter = letter
        self.normal = normal
        self.offset = offset
        self.start = start
        self.end = end
        self.camera = camera
        self.length = float(np.linalg.norm(end - start))


class RoomEdge:

    def __init__(self, start: tuple, end: tuple):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        direction = self.end - self.start
        self.length = float(np.linalg.norm(direction))
        self.direction = direction / self.length
        self.normal = (round(self.direction[1]), round(-self.direction[0]))
        self.wall_id = ""

    def line_offset(self) -> float:
        return float(np.array(self.normal) @ self.start)

    def along(self, point: np.ndarray) -> float:
        return float(self.direction @ (point - self.start))


class GeometryLayoutBuilder:

    def __init__(self):
        self.notes = []

    def rotate(self, point, angle: float) -> np.ndarray:
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        return np.array([cos_value * point[0] - sin_value * point[1], sin_value * point[0] + cos_value * point[1]])

    def snap_to_axis(self, normal: np.ndarray):
        best_axis = None
        best_dot = math.cos(math.radians(SNAP_ANGLE_DEG))

        for axis in AXES:
            dot = float(normal @ np.array(axis))
            if dot > best_dot:
                best_axis = axis
                best_dot = dot

        return best_axis

    def world_walls(self, photo_set: RoomPhotoGeometry, placements: RoomPlacements) -> list:
        placement_by_id = {}
        for placement in placements.placements:
            placement_by_id[placement.photo_id] = placement

        world_walls = []

        for photo in photo_set.photos:
            placement = placement_by_id[photo.photo_id]
            heading = math.radians(placement.heading_deg)
            camera = np.array(placement.position)

            for wall in photo.walls:
                if wall.visible_length_m < MIN_WALL_LENGTH_M:
                    continue

                direction = np.array(wall.end) - np.array(wall.start)
                direction = direction / np.linalg.norm(direction)
                normal = self.rotate(np.array([direction[1], -direction[0]]), heading)

                axis = self.snap_to_axis(normal)
                if axis is None:
                    continue

                start = camera + self.rotate(wall.start, heading)
                end = camera + self.rotate(wall.end, heading)
                offset = float(np.array(axis) @ (start + end) / 2)

                world_walls.append(WorldWall(photo.photo_id, wall.letter, axis, offset, start, end, camera,
                                             wall.start_type, wall.end_type))

        return world_walls

    def weighted_median(self, values: list, weights: list) -> float:
        order = np.argsort(values)
        sorted_values = np.array(values)[order]
        sorted_weights = np.array(weights)[order]
        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, cumulative[-1] / 2))
        return float(sorted_values[index])

    def offset_of(self, world_wall: WorldWall) -> float:
        return world_wall.offset

    def cluster_offsets(self, world_walls: list) -> list:
        clusters = []

        for axis in AXES:
            members = [world_wall for world_wall in world_walls if world_wall.normal == axis]
            members.sort(key=self.offset_of)

            group = []
            for member in members:
                if group and member.offset - group[-1].offset > CLUSTER_GAP_M:
                    clusters.append((axis, group))
                    group = []
                group.append(member)

            if group:
                clusters.append((axis, group))

        results = []
        for axis, group in clusters:
            offsets = [member.offset for member in group]
            weights = [member.length for member in group]
            results.append({"axis": axis, "offset": self.weighted_median(offsets, weights), "members": group})

        return results

    def grid_lines(self, clusters: list, use_end_lines: bool = True) -> tuple:
        xs = []
        ys = []

        end_xs = []
        end_ys = []

        for cluster in clusters:
            axis = cluster["axis"]
            ends = []
            if use_end_lines and self.total_length(cluster) >= MIN_END_LINE_LENGTH_M:
                ends = self.cluster_ends(cluster)

            if axis[1] == 0:
                xs.append(cluster["offset"] * axis[0])
                end_ys.extend(ends)
            else:
                ys.append(cluster["offset"] * axis[1])
                end_xs.extend(ends)

        xs = self.merge_close(xs)
        ys = self.merge_close(ys)

        return self.add_end_lines(xs, end_xs), self.add_end_lines(ys, end_ys)

    def add_end_lines(self, lines: list, end_lines: list) -> list:
        result = list(lines)

        for end_line in sorted(end_lines):
            if not result or end_line <= result[0] or end_line >= result[-1]:
                continue

            nearest = min(abs(end_line - line) for line in result)
            if nearest >= END_LINE_MIN_GAP_M:
                result.append(end_line)
                result.sort()

        return result

    def total_length(self, cluster: dict) -> float:
        return sum(member.length for member in cluster["members"])

    def cluster_ends(self, cluster: dict) -> list:
        along_index = 1 if cluster["axis"][1] == 0 else 0
        values = []

        for member in cluster["members"]:
            if member.start_type != "truncated":
                values.append(float(member.start[along_index]))
            if member.end_type != "truncated":
                values.append(float(member.end[along_index]))

        if not values:
            return []

        return [min(values), max(values)]

    def merge_close(self, values: list) -> list:
        merged = []

        for value in sorted(values):
            if merged and value - merged[-1][-1] < GRID_MERGE_M:
                merged[-1].append(value)
            else:
                merged.append([value])

        return [float(np.mean(group)) for group in merged]

    def cell_of(self, point: np.ndarray, xs: list, ys: list):
        if not (xs[0] < point[0] < xs[-1] and ys[0] < point[1] < ys[-1]):
            return None

        column = bisect.bisect_right(xs, point[0]) - 1
        row = bisect.bisect_right(ys, point[1]) - 1
        return column, row

    def carve_free_space(self, world_walls: list, xs: list, ys: list) -> np.ndarray:
        hits = np.zeros((len(xs) - 1, len(ys) - 1), dtype=int)

        for world_wall in world_walls:
            sample_count = max(int(world_wall.length / SAMPLE_STEP_M), 1) + 1

            for fraction in np.linspace(0, 1, sample_count):
                target = world_wall.start + fraction * (world_wall.end - world_wall.start)
                ray_length = float(np.linalg.norm(target - world_wall.camera))
                if ray_length <= RAY_STOP_SHORT_M:
                    continue

                step_count = max(int(ray_length / RAY_STEP_M), 1)
                last_step = (ray_length - RAY_STOP_SHORT_M) / ray_length

                seen_cells = set()
                for step in np.linspace(0, last_step, step_count):
                    cell = self.cell_of(world_wall.camera + step * (target - world_wall.camera), xs, ys)
                    if cell is not None:
                        seen_cells.add(cell)

                for cell in seen_cells:
                    hits[cell] += 1

        if not (hits > 0).any():
            return hits > 0

        typical = float(np.median(hits[hits > 0]))
        return hits >= max(MIN_CELL_HITS, MIN_HIT_SHARE * typical)

    def neighbours(self, cell: tuple, shape: tuple) -> list:
        column, row = cell
        result = []

        for step_column, step_row in AXES:
            other = (column + step_column, row + step_row)
            if 0 <= other[0] < shape[0] and 0 <= other[1] < shape[1]:
                result.append(other)

        return result

    def largest_region(self, inside: np.ndarray) -> np.ndarray:
        seen = np.zeros_like(inside)
        best = []

        for column in range(inside.shape[0]):
            for row in range(inside.shape[1]):
                if not inside[column, row] or seen[column, row]:
                    continue

                region = []
                stack = [(column, row)]
                seen[column, row] = True

                while stack:
                    cell = stack.pop()
                    region.append(cell)
                    for other in self.neighbours(cell, inside.shape):
                        if inside[other] and not seen[other]:
                            seen[other] = True
                            stack.append(other)

                if len(region) > len(best):
                    best = region

        result = np.zeros_like(inside)
        for cell in best:
            result[cell] = True

        return result

    def fill_holes(self, inside: np.ndarray) -> np.ndarray:
        outside = np.zeros_like(inside)
        stack = []

        for column in range(inside.shape[0]):
            for row in range(inside.shape[1]):
                on_border = column in (0, inside.shape[0] - 1) or row in (0, inside.shape[1] - 1)
                if on_border and not inside[column, row]:
                    outside[column, row] = True
                    stack.append((column, row))

        while stack:
            cell = stack.pop()
            for other in self.neighbours(cell, inside.shape):
                if not inside[other] and not outside[other]:
                    outside[other] = True
                    stack.append(other)

        return ~outside

    def is_inside(self, inside: np.ndarray, column: int, row: int) -> bool:
        if 0 <= column < inside.shape[0] and 0 <= row < inside.shape[1]:
            return bool(inside[column, row])
        return False

    def boundary_edges(self, inside: np.ndarray) -> dict:
        edges = {}

        for column in range(inside.shape[0]):
            for row in range(inside.shape[1]):
                if not inside[column, row]:
                    continue

                if not self.is_inside(inside, column, row + 1):
                    edges[(column, row + 1)] = (column + 1, row + 1)
                if not self.is_inside(inside, column + 1, row):
                    edges[(column + 1, row + 1)] = (column + 1, row)
                if not self.is_inside(inside, column, row - 1):
                    edges[(column + 1, row)] = (column, row)
                if not self.is_inside(inside, column - 1, row):
                    edges[(column, row)] = (column, row + 1)

        return edges

    def trace_outline(self, inside: np.ndarray) -> list:
        edges = self.boundary_edges(inside)
        start = max(edges.keys(), key=self.top_left_score)

        corners = [start]
        current = edges[start]
        while current != start:
            corners.append(current)
            current = edges[current]

        return self.remove_straight_points(corners)

    def top_left_score(self, vertex: tuple) -> tuple:
        return vertex[1], -vertex[0]

    def remove_straight_points(self, points: list) -> list:
        kept = []

        for index in range(len(points)):
            before = np.array(points[index - 1])
            here = np.array(points[index])
            after = np.array(points[(index + 1) % len(points)])

            first = here - before
            second = after - here
            if first[0] * second[1] - first[1] * second[0] != 0:
                kept.append(points[index])

        return kept

    def build_edges(self, corners: list, xs: list, ys: list) -> list:
        points = [(xs[column], ys[row]) for column, row in corners]

        edges = []
        for index in range(len(points)):
            edges.append(RoomEdge(points[index], points[(index + 1) % len(points)]))

        first_east = 0
        for index, edge in enumerate(edges):
            if edge.normal == (0, -1):
                first_east = index
                break

        edges = edges[first_east:] + edges[:first_east]
        for index, edge in enumerate(edges):
            edge.wall_id = f"W{index + 1}"

        return edges

    def corner_type(self, edge: RoomEdge, next_edge: RoomEdge) -> str:
        cross = edge.direction[0] * next_edge.direction[1] - edge.direction[1] * next_edge.direction[0]
        return "normal" if cross < 0 else "protruding"

    def match_wall(self, world_wall: WorldWall, edges: list):
        best_edge = None
        best_gap = MATCH_DISTANCE_M

        for edge in edges:
            if edge.normal != world_wall.normal:
                continue

            gap = abs(edge.line_offset() - world_wall.offset)
            low = min(edge.along(world_wall.start), edge.along(world_wall.end))
            high = max(edge.along(world_wall.start), edge.along(world_wall.end))
            overlaps = high > -MATCH_OVERLAP_M and low < edge.length + MATCH_OVERLAP_M

            if overlaps and gap < best_gap:
                best_edge = edge
                best_gap = gap

        return best_edge

    def opening_world_point(self, photo, opening, placement) -> np.ndarray:
        for wall in photo.walls:
            if wall.letter != opening.wall_letter:
                continue

            start = np.array(wall.start)
            direction = np.array(wall.end) - start
            direction = direction / np.linalg.norm(direction)
            local_point = start + (opening.start_along_wall_m + opening.width_m / 2) * direction

            heading = math.radians(placement.heading_deg)
            return np.array(placement.position) + self.rotate(local_point, heading)

        return None

    def match_openings(self, photo_set: RoomPhotoGeometry, placements: RoomPlacements, wall_matches: dict, edges: list) -> list:
        placement_by_id = {}
        for placement in placements.placements:
            placement_by_id[placement.photo_id] = placement

        edge_by_id = {}
        for edge in edges:
            edge_by_id[edge.wall_id] = edge

        found = []
        matches = []
        counters = {"door": 0, "window": 0}

        for photo in photo_set.photos:
            for opening in photo.openings:
                wall_id = wall_matches.get((photo.photo_id, opening.wall_letter))

                if wall_id is None or opening.width_m is None or opening.start_along_wall_m is None:
                    matches.append(OpeningMatch(photo_opening_id=opening.id, room_opening_id="none", type="none", room_wall_id="none"))
                    continue

                point = self.opening_world_point(photo, opening, placement_by_id[photo.photo_id])
                centre = edge_by_id[wall_id].along(point)

                room_opening_id = None
                for item in found:
                    if item["wall_id"] == wall_id and item["type"] == opening.type and abs(item["centre"] - centre) < SAME_OPENING_M:
                        room_opening_id = item["id"]

                if room_opening_id is None:
                    counters[opening.type] += 1
                    prefix = "D" if opening.type == "door" else "N"
                    room_opening_id = f"{prefix}{counters[opening.type]}"
                    found.append({"id": room_opening_id, "wall_id": wall_id, "type": opening.type, "centre": centre})

                matches.append(OpeningMatch(photo_opening_id=opening.id, room_opening_id=room_opening_id,
                                            type=opening.type, room_wall_id=wall_id))

        return matches

    def build_outline(self, world_walls: list, clusters: list, use_end_lines: bool) -> list:
        xs, ys = self.grid_lines(clusters, use_end_lines)

        if len(xs) < 2 or len(ys) < 2:
            raise ValueError("Not enough walls found in two directions to build a room outline")

        inside = self.carve_free_space(world_walls, xs, ys)
        inside = self.fill_holes(self.largest_region(inside))
        if not inside.any():
            raise ValueError("Could not find the room floor area between the walls")

        corners = self.trace_outline(inside)
        return self.build_edges(corners, xs, ys)

    def build_layout(self, photo_set: RoomPhotoGeometry, placements: RoomPlacements, room_type: str = "other") -> RoomLayout:
        self.notes = []

        world_walls = self.world_walls(photo_set, placements)
        clusters = self.cluster_offsets(world_walls)
        edges = self.build_outline(world_walls, clusters, True)

        if min(edge.length for edge in edges) < MIN_EDGE_LENGTH_M:
            self.notes.append("outline had tiny walls, rebuilt without wall-end lines")
            edges = self.build_outline(world_walls, clusters, False)

        wall_infos = []
        for index, edge in enumerate(edges):
            next_edge = edges[(index + 1) % len(edges)]
            wall_infos.append(RoomWallInfo(
                id=edge.wall_id,
                description=f"{edge.length:.2f} m wall, faces {AXIS_NAMES[edge.normal]}",
                corner_after=self.corner_type(edge, next_edge),
                position_m=round(edge.line_offset(), 3),
            ))

        wall_matches = {}
        unmatched = 0
        for world_wall in world_walls:
            edge = self.match_wall(world_wall, edges)
            if edge is None:
                unmatched += 1
            else:
                wall_matches[(world_wall.photo_id, world_wall.letter)] = edge.wall_id

        match_list = []
        for (photo_id, letter), wall_id in wall_matches.items():
            match_list.append(WallMatch(photo_id=photo_id, wall_letter=letter, room_wall_id=wall_id))

        opening_matches = self.match_openings(photo_set, placements, wall_matches, edges)

        self.notes.append(f"{len(world_walls)} photo walls, {len(clusters)} wall lines, "
                          f"{len(edges)} room walls, {unmatched} photo walls not on the outline")

        return RoomLayout(
            room_type=room_type,
            shape_description=f"{len(edges)} walls found from camera poses and wall geometry",
            walls=wall_infos,
            wall_matches=match_list,
            opening_matches=opening_matches,
            notes="; ".join(self.notes),
        )
