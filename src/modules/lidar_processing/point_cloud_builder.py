import logging

import numpy as np

from modules.lidar_processing.capture_reader import StrayCapture


VOXEL_SIZE_M = 0.02
MAX_DEPTH_M = 4.5
MIN_DEPTH_M = 0.2
MIN_CONFIDENCE = 2
NORMAL_STEP_PX = 2
MAX_DEPTH_JUMP = 0.05
FRAMES_PER_BATCH = 60
MIN_VOXEL_POINTS = 2
MAX_FUSED_FRAMES = 900
KEY_OFFSET = 1 << 20

logger = logging.getLogger(__name__)


class PointCloud:

    def __init__(self, points: np.ndarray, normals: np.ndarray, counts: np.ndarray, camera_path: np.ndarray):
        self.points = points
        self.normals = normals
        self.counts = counts
        self.camera_path = camera_path

    def save(self, path: str):
        np.savez_compressed(path, points=self.points, normals=self.normals, counts=self.counts, camera_path=self.camera_path)

    def save_ply(self, path: str):
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(self.points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "end_header\n"
        )
        rows = np.hstack([self.points, self.normals]).astype("<f4")

        with open(path, "wb") as ply_file:
            ply_file.write(header.encode("ascii"))
            ply_file.write(rows.tobytes())


class PointCloudLoader:

    def load(self, path: str) -> PointCloud:
        data = np.load(path)
        return PointCloud(data["points"], data["normals"], data["counts"], data["camera_path"])


class VoxelGrid:

    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size
        self.keys = np.zeros(0, dtype=np.int64)
        self.point_sums = np.zeros((0, 3))
        self.normal_sums = np.zeros((0, 3))
        self.counts = np.zeros(0, dtype=np.int64)

    def voxel_keys(self, points: np.ndarray) -> np.ndarray:
        cells = np.floor(points / self.voxel_size).astype(np.int64) + KEY_OFFSET
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    def merge(self, keys: np.ndarray, point_sums: np.ndarray, normal_sums: np.ndarray, counts: np.ndarray):
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        merged_points = np.zeros((len(unique_keys), 3))
        merged_normals = np.zeros((len(unique_keys), 3))

        for axis in range(3):
            merged_points[:, axis] = np.bincount(inverse, weights=point_sums[:, axis], minlength=len(unique_keys))
            merged_normals[:, axis] = np.bincount(inverse, weights=normal_sums[:, axis], minlength=len(unique_keys))

        merged_counts = np.bincount(inverse, weights=counts, minlength=len(unique_keys)).astype(np.int64)
        return unique_keys, merged_points, merged_normals, merged_counts

    def add(self, points: np.ndarray, normals: np.ndarray):
        if len(points) == 0:
            return

        keys, point_sums, normal_sums, counts = self.merge(self.voxel_keys(points), points, normals, np.ones(len(points)))

        all_keys = np.concatenate([self.keys, keys])
        all_points = np.concatenate([self.point_sums, point_sums])
        all_normals = np.concatenate([self.normal_sums, normal_sums])
        all_counts = np.concatenate([self.counts, counts])

        self.keys, self.point_sums, self.normal_sums, self.counts = self.merge(all_keys, all_points, all_normals, all_counts)

    def to_points(self, min_points: int) -> tuple:
        keep = self.counts >= min_points
        counts = self.counts[keep]
        points = self.point_sums[keep] / counts[:, None]

        normals = self.normal_sums[keep]
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(lengths, 1e-9)[:, None]

        return points.astype(np.float32), normals.astype(np.float32), counts


class PointCloudBuilder:

    def __init__(self, every_nth_frame: int = 3):
        self.every_nth_frame = every_nth_frame

    def frame_step(self, frame_count: int) -> int:
        return max(self.every_nth_frame, int(np.ceil(frame_count / MAX_FUSED_FRAMES)))

    def pixel_rays(self, capture: StrayCapture) -> np.ndarray:
        intrinsics = capture.depth_intrinsics()
        cols, rows = np.meshgrid(np.arange(capture.depth_width), np.arange(capture.depth_height))
        x = (cols + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0]
        y = (rows + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1]
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def camera_normals(self, points: np.ndarray, depth: np.ndarray) -> tuple:
        step = NORMAL_STEP_PX
        across = np.zeros_like(points)
        down = np.zeros_like(points)
        across[:, step:-step] = points[:, 2 * step:] - points[:, :-2 * step]
        down[step:-step, :] = points[2 * step:, :] - points[:-2 * step, :]

        normals = np.cross(across, down)
        lengths = np.linalg.norm(normals, axis=-1)
        normals = normals / np.maximum(lengths, 1e-9)[..., None]

        facing = np.sum(normals * points, axis=-1)
        normals[facing > 0] = -normals[facing > 0]

        jump = np.zeros(depth.shape, dtype=bool)
        jump[:, step:-step] |= np.abs(depth[:, 2 * step:] - depth[:, :-2 * step]) > MAX_DEPTH_JUMP * depth[:, step:-step]
        jump[step:-step, :] |= np.abs(depth[2 * step:, :] - depth[:-2 * step, :]) > MAX_DEPTH_JUMP * depth[step:-step, :]

        border = np.zeros(depth.shape, dtype=bool)
        border[step:-step, step:-step] = True

        valid = (lengths > 1e-9) & border & ~jump
        return normals, valid

    def frame_points(self, capture: StrayCapture, frame_index: int, rays: np.ndarray) -> tuple:
        file_name = capture.depth_files[frame_index]
        depth = capture.read_depth(file_name)
        confidence = capture.read_confidence(file_name)

        points = rays * depth[..., None]
        normals, valid = self.camera_normals(points, depth)

        valid &= (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M)
        if confidence is not None:
            valid &= confidence >= MIN_CONFIDENCE

        pose = capture.poses[frame_index]
        world_points = pose.to_world(points[valid])
        world_normals = pose.direction_to_world(normals[valid])
        return world_points, world_normals

    def build(self, capture: StrayCapture) -> PointCloud:
        rays = self.pixel_rays(capture)
        grid = VoxelGrid(VOXEL_SIZE_M)
        all_indexes = capture.frame_indexes()
        frame_indexes = all_indexes[::self.frame_step(len(all_indexes))]

        batch_points = []
        batch_normals = []

        for count, frame_index in enumerate(frame_indexes):
            points, normals = self.frame_points(capture, frame_index, rays)
            batch_points.append(points)
            batch_normals.append(normals)

            if len(batch_points) == FRAMES_PER_BATCH or count == len(frame_indexes) - 1:
                grid.add(np.concatenate(batch_points), np.concatenate(batch_normals))
                batch_points = []
                batch_normals = []
                logger.info("Fused %d of %d depth frames (%d voxels)", count + 1, len(frame_indexes), len(grid.keys))

        points, normals, counts = grid.to_points(MIN_VOXEL_POINTS)
        return PointCloud(points, normals, counts, capture.camera_path().astype(np.float32))
