import numpy as np
from scipy import ndimage

from modules.image_processing.geometry_layout import GeometryLayoutBuilder
from modules.lidar_processing.plan_grid import CELL_SIZE_M, PlanGrid


PEAK_BIN_M = 0.01
PEAK_WINDOW_M = 0.03
MIN_PEAK_POINTS = 30
TALL_EXTENT_M = 0.8
MIN_TALL_LENGTH_M = 0.4
SAME_LINE_M = 0.12
SMALL_GAP_M = 0.15
MIN_DOOR_GAP_M = 0.55
MAX_DOOR_GAP_M = 1.05
MIN_ROOM_AREA_M2 = 0.8
MIN_RUN_M = 0.3
CLOSE_GAP_M = 1.1
SNAP_DISTANCE_M = 0.25
VISIT_REACH_M = 0.3
CLOSET_AREA_M2 = 3.0
CLOSET_REACH_M = 1.5
NOTCH_FILL_PASSES = 2
RASTER_SIGMA_M = 0.05
MIN_LINE_SIGMA_M = 0.005


class WallLine:

    def __init__(self, axis: int, position: float, sign: int, tall_length: float, spread: float, count: int):
        self.axis = axis
        self.position = position
        self.sign = sign
        self.tall_length = tall_length
        self.spread = spread
        self.count = count


class LidarRoom:

    def __init__(self, index: int, mask: np.ndarray, edges: list, area: float, sigmas: dict):
        self.index = index
        self.mask = mask
        self.edges = edges
        self.area = area
        self.sigmas = sigmas

    def sigma_of(self, axis_name: str, value: float) -> float:
        return self.sigmas.get((axis_name, round(value, 6)), RASTER_SIGMA_M)


class WallLineFinder:

    def tall_length(self, along: np.ndarray, heights: np.ndarray) -> float:
        bins = np.floor(along / CELL_SIZE_M).astype(int)
        order = np.argsort(bins)
        bins = bins[order]
        heights = heights[order]

        starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
        lows = np.minimum.reduceat(heights, starts)
        highs = np.maximum.reduceat(heights, starts)
        return float(np.sum(highs - lows >= TALL_EXTENT_M) * CELL_SIZE_M)

    def strength_of(self, line: WallLine) -> float:
        return line.tall_length

    def find_for_sign(self, axis: int, values: np.ndarray, along: np.ndarray, heights: np.ndarray, sign: int) -> list:
        if len(values) < MIN_PEAK_POINTS:
            return []

        edges = np.arange(values.min() - 0.05, values.max() + 0.05 + PEAK_BIN_M, PEAK_BIN_M)
        counts, edges = np.histogram(values, bins=edges)
        smooth = np.convolve(counts, [1, 2, 3, 2, 1], mode="same") / 9.0

        lines = []
        for index in range(1, len(smooth) - 1):
            if not (smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]):
                continue

            centre = (edges[index] + edges[index + 1]) / 2
            members = np.abs(values - centre) < PEAK_WINDOW_M
            if members.sum() < MIN_PEAK_POINTS:
                continue

            tall_length = self.tall_length(along[members], heights[members])
            if tall_length < MIN_TALL_LENGTH_M:
                continue

            position = float(np.median(values[members]))
            spread = float(np.std(values[members]))
            lines.append(WallLine(axis, position, sign, tall_length, spread, int(members.sum())))

        kept = []
        for line in sorted(lines, key=self.strength_of, reverse=True):
            is_close = False
            for other in kept:
                if abs(other.position - line.position) < SAME_LINE_M:
                    is_close = True
            if not is_close:
                kept.append(line)

        return kept

    def find(self, grid: PlanGrid, axis: int) -> list:
        points = grid.wall_points[axis]
        heights = grid.wall_heights[axis]
        signs = grid.wall_signs[axis]

        lines = []
        for sign in (1, -1):
            chosen = signs == sign
            lines.extend(self.find_for_sign(axis, points[chosen, axis], points[chosen, 1 - axis], heights[chosen], sign))

        return lines


class RoomSegmenter:

    def __init__(self):
        self.layout_builder = GeometryLayoutBuilder()
        self.notes = []

    def evidence_along(self, grid: PlanGrid, line: WallLine) -> np.ndarray:
        other = 1 - line.axis
        start = grid.origin[other]
        end = grid.origin[other] + grid.shape[other] * CELL_SIZE_M
        return grid.wall_extent_along(line.axis, line.position, start, end) >= TALL_EXTENT_M

    def runs(self, evidence: np.ndarray) -> list:
        filled = evidence.copy()
        gap_cells = int(round(SMALL_GAP_M / CELL_SIZE_M))
        on = np.flatnonzero(evidence)

        for index in range(len(on) - 1):
            if 1 < on[index + 1] - on[index] <= gap_cells + 1:
                filled[on[index]:on[index + 1]] = True

        runs = []
        index = 0
        while index < len(filled):
            if filled[index]:
                first = index
                while index < len(filled) and filled[index]:
                    index += 1
                runs.append((first, index - 1))
            else:
                index += 1

        return runs

    def find_door_gaps(self, grid: PlanGrid, lines: list) -> list:
        gaps = []

        for line in lines:
            runs = self.runs(self.evidence_along(grid, line))
            other = 1 - line.axis

            for index in range(len(runs) - 1):
                first_cell = runs[index][1] + 1
                last_cell = runs[index + 1][0] - 1
                width = (last_cell - first_cell + 1) * CELL_SIZE_M

                if MIN_DOOR_GAP_M <= width <= MAX_DOOR_GAP_M:
                    gaps.append({
                        "axis": line.axis,
                        "position": line.position,
                        "sign": line.sign,
                        "start": grid.to_metres(first_cell, other) - CELL_SIZE_M / 2,
                        "end": grid.to_metres(last_cell, other) + CELL_SIZE_M / 2,
                        "first_cell": first_cell,
                        "last_cell": last_cell,
                    })

        return gaps

    def line_barrier(self, grid: PlanGrid, lines: list, axis: int) -> np.ndarray:
        barrier = np.zeros(grid.shape, dtype=bool)

        for line in lines:
            if line.axis != axis:
                continue

            line_cell = grid.to_cell(line.position, line.axis)
            if not 0 <= line_cell < grid.shape[line.axis]:
                continue

            for first, last in self.runs(self.evidence_along(grid, line)):
                if (last - first + 1) * CELL_SIZE_M < MIN_RUN_M:
                    continue
                if line.axis == 0:
                    barrier[line_cell, first:last + 1] = True
                else:
                    barrier[first:last + 1, line_cell] = True

        return barrier

    def barrier_map(self, grid: PlanGrid, lines: list, door_gaps: list) -> np.ndarray:
        barrier = np.zeros(grid.shape, dtype=bool)
        length = int(round(CLOSE_GAP_M / CELL_SIZE_M))
        along_wall = [np.ones((1, length), dtype=bool), np.ones((length, 1), dtype=bool)]

        for axis in range(2):
            extent = grid.wall_high[axis] - grid.wall_low[axis]
            facing = self.line_barrier(grid, lines, axis) | (np.isfinite(extent) & (extent >= TALL_EXTENT_M))
            barrier |= facing | ndimage.binary_closing(facing, structure=along_wall[axis])

        for gap in door_gaps:
            line_cell = grid.to_cell(gap["position"], gap["axis"])
            low = max(line_cell - 1, 0)
            high = min(line_cell + 2, grid.shape[gap["axis"]])

            if gap["axis"] == 0:
                barrier[low:high, gap["first_cell"]:gap["last_cell"] + 1] = True
            else:
                barrier[gap["first_cell"]:gap["last_cell"] + 1, low:high] = True

        return barrier


    def room_masks(self, labels: np.ndarray) -> list:
        sizes = np.bincount(labels.ravel())
        min_cells = MIN_ROOM_AREA_M2 / CELL_SIZE_M ** 2

        masks = []
        for label in range(1, len(sizes)):
            if sizes[label] >= min_cells:
                masks.append(labels == label)

        return masks

    def visited_map(self, grid: PlanGrid, reach_m: float) -> np.ndarray:
        visited = np.zeros(grid.shape, dtype=bool)
        cells = grid.cell_index(grid.camera_path)
        visited[cells[:, 0], cells[:, 1]] = True

        reach_cells = int(round(reach_m / CELL_SIZE_M))
        return ndimage.binary_dilation(visited, iterations=reach_cells)

    def was_visited(self, mask: np.ndarray, near_camera: np.ndarray, reachable: np.ndarray) -> bool:
        area = mask.sum() * CELL_SIZE_M ** 2
        if area < CLOSET_AREA_M2:
            return bool((mask & reachable).any())
        return bool((mask & near_camera).any())

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool), border_value=0)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))

        labels, count = ndimage.label(mask)
        if count == 0:
            return mask

        biggest = int(np.bincount(labels.ravel())[1:].argmax()) + 1
        return ndimage.binary_fill_holes(labels == biggest)

    def room_seen_from(self, mask: np.ndarray, barrier: np.ndarray, axis: int, forward: bool) -> np.ndarray:
        seen = np.zeros(mask.shape, dtype=bool)
        state = np.zeros(mask.shape[1 - axis], dtype=bool)
        indexes = range(mask.shape[axis]) if forward else range(mask.shape[axis] - 1, -1, -1)

        for index in indexes:
            if axis == 0:
                seen[index, :] = state
                state = np.where(mask[index, :], True, np.where(barrier[index, :], False, state))
            else:
                seen[:, index] = state
                state = np.where(mask[:, index], True, np.where(barrier[:, index], False, state))

        return seen

    def fill_notches(self, mask: np.ndarray, barrier: np.ndarray, blocked: np.ndarray) -> np.ndarray:
        for repeat in range(NOTCH_FILL_PASSES):
            votes = np.zeros(mask.shape, dtype=int)
            for axis in range(2):
                for forward in (True, False):
                    votes += self.room_seen_from(mask, barrier, axis, forward)

            mask = mask | ((votes >= 3) & ~barrier & ~blocked)

        return mask

    def pixel_outline(self, grid: PlanGrid, mask: np.ndarray) -> list:
        padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = mask

        corners = self.layout_builder.trace_outline(padded)

        points = []
        for column, row in corners:
            x = grid.origin[0] + (column - 1) * CELL_SIZE_M
            y = grid.origin[1] + (row - 1) * CELL_SIZE_M
            points.append((x, y))

        return points

    def snap_value(self, value: float, lines: list, inside_sign: int) -> float:
        best = None
        best_score = None

        for line in lines:
            gap = abs(line.position - value)
            if gap > SNAP_DISTANCE_M:
                continue

            score = gap if line.sign == inside_sign else gap + SNAP_DISTANCE_M
            if best_score is None or score < best_score:
                best = line
                best_score = score

        if best is None:
            return value, RASTER_SIGMA_M
        return best.position, best.spread / np.sqrt(best.count) + MIN_LINE_SIGMA_M

    def snap_outline(self, points: list, x_lines: list, y_lines: list, sigmas: dict) -> list:
        count = len(points)
        edge_values = []

        for index in range(count):
            start = points[index]
            end = points[(index + 1) % count]
            direction = (np.sign(end[0] - start[0]), np.sign(end[1] - start[1]))
            inside = (direction[1], -direction[0])

            if direction[0] == 0:
                value, sigma = self.snap_value(start[0], x_lines, int(inside[0]))
                edge_values.append(("x", value))
            else:
                value, sigma = self.snap_value(start[1], y_lines, int(inside[1]))
                edge_values.append(("y", value))

            key = (edge_values[-1][0], round(value, 6))
            sigmas[key] = min(sigmas.get(key, sigma), sigma)

        snapped = []
        for index in range(count):
            before = edge_values[index - 1]
            after = edge_values[index]
            if before[0] == "x":
                snapped.append((before[1], after[1]))
            else:
                snapped.append((after[1], before[1]))

        return snapped

    def outline_edges(self, points: list) -> list:
        xs = sorted(set(point[0] for point in points))
        ys = sorted(set(point[1] for point in points))

        corners = []
        for x, y in points:
            corners.append((xs.index(x), ys.index(y)))

        corners = self.layout_builder.clean_corners(corners)
        corners = self.layout_builder.simplify_outline(corners, xs, ys)
        corners = self.layout_builder.clean_corners(corners)
        return self.layout_builder.build_edges(corners, xs, ys)

    def polygon_area(self, edges: list) -> float:
        points = np.array([edge.start for edge in edges])
        x = points[:, 0]
        y = points[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)

    def area_of(self, room: LidarRoom) -> float:
        return room.area

    def segment(self, grid: PlanGrid, x_lines: list, y_lines: list) -> dict:
        self.notes = []
        self.layout_builder.notes = []

        if len(x_lines) < 2 or len(y_lines) < 2:
            raise ValueError("Not enough walls found in the LiDAR scan to build a floor plan")

        door_gaps = self.find_door_gaps(grid, x_lines + y_lines)
        barrier = self.barrier_map(grid, x_lines + y_lines, door_gaps)
        labels, count = ndimage.label(grid.free & ~barrier)
        near_camera = self.visited_map(grid, VISIT_REACH_M)
        reachable = self.visited_map(grid, CLOSET_REACH_M)

        masks = []
        skipped = 0
        for mask in self.room_masks(labels):
            if self.was_visited(mask, near_camera, reachable):
                masks.append(mask)
            else:
                skipped += 1

        taken = np.zeros(grid.shape, dtype=bool)
        for mask in masks:
            taken |= mask

        rooms = []
        for mask in masks:
            mask = self.clean_mask(self.fill_notches(mask, barrier, taken & ~mask))
            sigmas = {}
            points = self.snap_outline(self.pixel_outline(grid, mask), x_lines, y_lines, sigmas)
            edges = self.outline_edges(points)
            if len(edges) < 4:
                continue

            rooms.append(LidarRoom(0, mask, edges, self.polygon_area(edges), sigmas))

        if not rooms:
            raise ValueError("No rooms found in the LiDAR scan")

        rooms.sort(key=self.area_of, reverse=True)
        for index, room in enumerate(rooms):
            room.index = index + 1

        if skipped:
            self.notes.append(f"skipped {skipped} areas the camera never entered (seen through windows or doors)")

        self.notes.extend(self.layout_builder.notes)
        return {"rooms": rooms, "labels": labels, "door_gaps": door_gaps}


class RoomFinder:

    def __init__(self, rooms: list):
        self.rooms = rooms

    def point_in_polygon(self, point: np.ndarray, edges: list) -> bool:
        inside = False

        for edge in edges:
            x1, y1 = edge.start
            x2, y2 = edge.end
            if (y1 > point[1]) != (y2 > point[1]) and point[0] < (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1:
                inside = not inside

        return inside

    def room_behind(self, room: LidarRoom, edge, along: float, reach: float):
        probe = edge.start + edge.direction * along - np.array(edge.normal) * reach

        for other in self.rooms:
            if other.index != room.index and self.point_in_polygon(probe, other.edges):
                return other
        return None

    def room_by_index(self, index: int) -> LidarRoom:
        for room in self.rooms:
            if room.index == index:
                return room
        return None

    def edge_by_id(self, room: LidarRoom, wall_id: str):
        for edge in room.edges:
            if edge.wall_id == wall_id:
                return edge
        return None
