import numpy as np
from scipy import ndimage

from modules.lidar_processing.plan_frame import HORIZONTAL_NORMAL, VERTICAL_NORMAL, PlanFrame
from modules.lidar_processing.point_cloud_builder import PointCloud


CELL_SIZE_M = 0.05
FLOOR_TOLERANCE_M = 0.08
LOW_BAND_M = 0.15
HIGH_BAND_M = 2.0
DEFAULT_TOP_M = 2.3
AXIS_NORMAL = 0.85
CLOSE_RADIUS_CELLS = 2
MARGIN_M = 0.5


class PlanGrid:

    def __init__(self, cloud: PointCloud, frame: PlanFrame):
        self.frame = frame

        plan_points = frame.to_plan(cloud.points)
        heights = frame.height(cloud.points)
        plan_normals = frame.direction_to_plan(cloud.normals)
        up = cloud.normals[:, 1]

        top = DEFAULT_TOP_M if frame.ceiling_height_m is None else frame.ceiling_height_m - 0.1
        in_room = (heights > -FLOOR_TOLERANCE_M) & (heights < top)

        low = np.percentile(plan_points[in_room], 0.2, axis=0) - MARGIN_M
        high = np.percentile(plan_points[in_room], 99.8, axis=0) + MARGIN_M
        self.origin = low
        self.shape = tuple((np.ceil((high - low) / CELL_SIZE_M)).astype(int) + 1)

        is_floor = (up > HORIZONTAL_NORMAL) & (np.abs(heights) < FLOOR_TOLERANCE_M)
        is_flat_thing = (np.abs(up) >= VERTICAL_NORMAL) & (heights > FLOOR_TOLERANCE_M) & (heights < top)
        self.floor = self.count_map(plan_points[is_floor]) > 0
        self.free = self.close(self.count_map(plan_points[is_floor | is_flat_thing]) > 0)

        is_wall = (np.abs(up) < VERTICAL_NORMAL) & (heights > LOW_BAND_M) & (heights < HIGH_BAND_M)
        self.wall_points = []
        self.wall_heights = []
        self.wall_signs = []
        self.wall_low = []
        self.wall_high = []

        for axis in range(2):
            facing = is_wall & (np.abs(plan_normals[:, axis]) > AXIS_NORMAL)
            self.wall_points.append(plan_points[facing])
            self.wall_heights.append(heights[facing])
            self.wall_signs.append(np.sign(plan_normals[facing, axis]))
            low_map, high_map = self.height_maps(plan_points[facing], heights[facing])
            self.wall_low.append(low_map)
            self.wall_high.append(high_map)

        self.any_wall = self.count_map(plan_points[is_wall]) > 0
        self.camera_path = frame.to_plan(cloud.camera_path)

    def cell_index(self, plan_points: np.ndarray) -> np.ndarray:
        index = np.floor((plan_points - self.origin) / CELL_SIZE_M).astype(int)
        return np.clip(index, 0, np.array(self.shape) - 1)

    def count_map(self, plan_points: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.shape, dtype=np.int32)
        index = self.cell_index(plan_points)
        np.add.at(counts, (index[:, 0], index[:, 1]), 1)
        return counts

    def height_maps(self, plan_points: np.ndarray, heights: np.ndarray) -> tuple:
        low_map = np.full(self.shape, np.inf, dtype=np.float32)
        high_map = np.full(self.shape, -np.inf, dtype=np.float32)
        index = self.cell_index(plan_points)
        np.minimum.at(low_map, (index[:, 0], index[:, 1]), heights)
        np.maximum.at(high_map, (index[:, 0], index[:, 1]), heights)
        return low_map, high_map

    def close(self, mask: np.ndarray) -> np.ndarray:
        size = 2 * CLOSE_RADIUS_CELLS + 1
        structure = np.ones((size, size), dtype=bool)
        return ndimage.binary_closing(mask, structure=structure, border_value=0)

    def to_cell(self, value: float, axis: int) -> int:
        return int(np.floor((value - self.origin[axis]) / CELL_SIZE_M))

    def to_metres(self, index: float, axis: int) -> float:
        return float(self.origin[axis] + (index + 0.5) * CELL_SIZE_M)

    def free_share(self, x_low: float, x_high: float, y_low: float, y_high: float) -> float:
        column_low = max(self.to_cell(x_low, 0) + 1, 0)
        column_high = min(self.to_cell(x_high, 0), self.shape[0])
        row_low = max(self.to_cell(y_low, 1) + 1, 0)
        row_high = min(self.to_cell(y_high, 1), self.shape[1])

        if column_high <= column_low or row_high <= row_low:
            column = min(max(self.to_cell((x_low + x_high) / 2, 0), 0), self.shape[0] - 1)
            row = min(max(self.to_cell((y_low + y_high) / 2, 1), 0), self.shape[1] - 1)
            return float(self.free[column, row])

        return float(self.free[column_low:column_high, row_low:row_high].mean())

    def wall_extent_along(self, axis: int, position: float, start: float, end: float, reach_cells: int = 1) -> np.ndarray:
        line_cell = self.to_cell(position, axis)
        other = 1 - axis
        first = max(self.to_cell(start, other), 0)
        last = min(self.to_cell(end, other), self.shape[other] - 1)
        if last < first:
            return np.zeros(0)

        low_cell = max(line_cell - reach_cells, 0)
        high_cell = min(line_cell + reach_cells + 1, self.shape[axis])

        if axis == 0:
            low_values = self.wall_low[0][low_cell:high_cell, first:last + 1].min(axis=0)
            high_values = self.wall_high[0][low_cell:high_cell, first:last + 1].max(axis=0)
        else:
            low_values = self.wall_low[1][first:last + 1, low_cell:high_cell].min(axis=1)
            high_values = self.wall_high[1][first:last + 1, low_cell:high_cell].max(axis=1)

        extent = high_values - low_values
        extent[~np.isfinite(extent)] = 0
        return extent
