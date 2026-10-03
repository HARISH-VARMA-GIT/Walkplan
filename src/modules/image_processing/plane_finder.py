import numpy as np
import open3d as o3d


CAMERA_UP = np.array([0.0, -1.0, 0.0])
FLOOR_MAX_TILT_DEG = 35
CEILING_MAX_TILT_DEG = 15
WALL_MAX_TILT_DEG = 15


class Plane:

    def __init__(self, normal: np.ndarray, offset: float, rows: np.ndarray, cols: np.ndarray, points: np.ndarray):
        self.normal = normal
        self.offset = offset
        self.rows = rows
        self.cols = cols
        self.points = points

    def distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset

    def point_count(self) -> int:
        return len(self.points)


class PlaneFinder:

    def __init__(self, distance_threshold=0.03, min_fraction=0.02, max_planes=12, step=4, iterations=1000):
        self.distance_threshold = distance_threshold
        self.min_fraction = min_fraction
        self.max_planes = max_planes
        self.step = step
        self.iterations = iterations

    def sample_points(self, points: np.ndarray, mask: np.ndarray):
        height, width = mask.shape
        rows, cols = np.mgrid[0:height:self.step, 0:width:self.step]
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)

        sampled = points[rows, cols]
        depth = sampled[:, 2]
        valid = mask[rows, cols] & np.isfinite(sampled).all(axis=1) & (depth > 0.2) & (depth < 15)

        return sampled[valid], rows[valid], cols[valid]

    def fit_plane(self, points: np.ndarray):
        centre = points.mean(axis=0)
        normal = np.linalg.svd(points - centre, full_matrices=False)[2][-1]
        offset = -float(normal @ centre)

        if offset < 0:
            normal = -normal
            offset = -offset

        return normal, offset

    def find_planes(self, points: np.ndarray, mask: np.ndarray) -> list:
        all_points, all_rows, all_cols = self.sample_points(points, mask)
        min_count = max(int(self.min_fraction * len(all_points)), 50)

        remaining = np.arange(len(all_points))
        planes = []

        while len(remaining) > min_count and len(planes) < self.max_planes:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(all_points[remaining].astype(np.float64))

            _, inlier_list = cloud.segment_plane(self.distance_threshold, 3, self.iterations)
            if len(inlier_list) < min_count:
                break

            inliers = remaining[np.array(inlier_list)]
            normal, offset = self.fit_plane(all_points[inliers])

            close = np.abs(all_points[remaining] @ normal + offset) < self.distance_threshold
            inliers = remaining[close]

            planes.append(Plane(normal, offset, all_rows[inliers], all_cols[inliers], all_points[inliers]))
            remaining = remaining[~close]

        return planes


class Surfaces:

    def __init__(self, up: np.ndarray, floor, ceiling, walls: list):
        self.up = up
        self.floor = floor
        self.ceiling = ceiling
        self.walls = walls


class SurfaceClassifier:

    def classify(self, planes: list) -> Surfaces:
        floor_limit = np.cos(np.radians(FLOOR_MAX_TILT_DEG))
        floor_candidates = []

        for plane in planes:
            if plane.normal @ CAMERA_UP > floor_limit:
                floor_candidates.append(plane)

        floor = self.pick_farthest(floor_candidates)
        up = CAMERA_UP if floor is None else floor.normal

        ceiling_limit = np.cos(np.radians(CEILING_MAX_TILT_DEG))
        wall_limit = np.sin(np.radians(WALL_MAX_TILT_DEG))
        ceiling_candidates = []
        walls = []

        for plane in planes:
            if plane is floor:
                continue

            alignment = plane.normal @ up

            if alignment < -ceiling_limit:
                ceiling_candidates.append(plane)
            elif abs(alignment) < wall_limit:
                walls.append(plane)

        ceiling = self.pick_farthest(ceiling_candidates)

        return Surfaces(up, floor, ceiling, walls)

    def pick_farthest(self, planes: list):
        best = None

        for plane in planes:
            if best is None or plane.offset > best.offset:
                best = plane

        return best
