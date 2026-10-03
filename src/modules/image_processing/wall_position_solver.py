import math

import numpy as np


BASE_SIGMA_M = 0.05
SIGMA_PER_METRE = 0.06
OUTLIER_LIMIT = 3.0
MIN_OUTLIER_M = 0.4
HEADING_LIMIT_DEG = 30
INSIDE_ROUNDS = 5
PRIOR_SIGMA_M = 0.3
WALL_PRIOR_SIGMA_M = 0.5


class PhotoPose:

    def __init__(self, photo_id: str, heading: float):
        self.photo_id = photo_id
        self.heading = heading
        self.position = None
        self.prior = None

    def rotate(self, point) -> np.ndarray:
        cos_value = math.cos(self.heading)
        sin_value = math.sin(self.heading)
        x, y = point
        return np.array([cos_value * x - sin_value * y, sin_value * x + cos_value * y])

    def to_room(self, point) -> np.ndarray:
        return self.position + self.rotate(point)


class Observation:

    def __init__(self, photo_id: str, wall_index: int, distance: float, kind: str = "distance"):
        self.photo_id = photo_id
        self.wall_index = wall_index
        self.distance = distance
        self.kind = kind
        self.point = None
        self.sigma = BASE_SIGMA_M + SIGMA_PER_METRE * abs(distance)
        self.active = True


class WallPositionSolver:

    def __init__(self, directions: list):
        self.directions = [np.array(direction, dtype=float) for direction in directions]
        self.normals = []
        for direction in self.directions:
            self.normals.append(np.array([direction[1], -direction[0]]))

        self.wall_count = len(self.directions)
        self.poses = {}
        self.observations = []
        self.notes = []
        self.wall_positions = None
        self.wall_sigmas = None
        self.wall_priors = {}

    def set_wall_prior(self, wall_index: int, position: float):
        self.wall_priors[wall_index] = position

    def angle_of(self, vector) -> float:
        return math.atan2(vector[1], vector[0])

    def heading_difference_deg(self, angle_a: float, angle_b: float) -> float:
        return math.degrees(abs(math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))))

    def estimate_heading(self, photo_id: str, seen_walls: list):
        angles = []
        for seen in seen_walls:
            angles.append(self.angle_of(self.normals[seen["wall_index"]]) - self.angle_of(seen["normal"]))

        heading = math.atan2(sum(math.sin(angle) for angle in angles), sum(math.cos(angle) for angle in angles))

        kept = []
        for seen, angle in zip(seen_walls, angles):
            if self.heading_difference_deg(angle, heading) < HEADING_LIMIT_DEG:
                kept.append(seen)
            else:
                self.notes.append(f"{photo_id}: wall match to W{seen['wall_index'] + 1} disagrees with the other walls, ignored")

        return heading, kept

    def has_two_directions(self, seen_walls: list) -> bool:
        for seen_a in seen_walls:
            for seen_b in seen_walls:
                if abs(self.normals[seen_a["wall_index"]] @ self.normals[seen_b["wall_index"]]) < 0.5:
                    return True
        return False

    def observed_walls(self) -> set:
        observed = set()
        for observation in self.observations:
            if observation.kind == "distance" and observation.active:
                observed.add(observation.wall_index)
        return observed

    def keep_walls_near_heading(self, photo_id: str, seen_walls: list, heading: float) -> list:
        kept = []

        for seen in seen_walls:
            angle = self.angle_of(self.normals[seen["wall_index"]]) - self.angle_of(seen["normal"])
            if self.heading_difference_deg(angle, heading) < HEADING_LIMIT_DEG:
                kept.append(seen)
            else:
                self.notes.append(f"{photo_id}: wall match to W{seen['wall_index'] + 1} disagrees with the camera pose, ignored")

        return kept

    def add_photo(self, photo_id: str, seen_walls: list, heading: float = None, prior_position=None):
        if not seen_walls:
            return

        if heading is None:
            heading, kept = self.estimate_heading(photo_id, seen_walls)
        else:
            kept = self.keep_walls_near_heading(photo_id, seen_walls, heading)

        if prior_position is None and not self.has_two_directions(kept):
            return
        if not kept:
            return

        self.poses[photo_id] = PhotoPose(photo_id, heading)
        self.poses[photo_id].prior = None if prior_position is None else np.array(prior_position, dtype=float)

        for seen in kept:
            self.observations.append(Observation(photo_id, seen["wall_index"], seen["distance"]))

            for point in seen["end_points"]:
                inside = Observation(photo_id, seen["wall_index"], 0.0, kind="inside")
                inside.point = point
                self.observations.append(inside)

    def unknown_count(self) -> int:
        return self.wall_count + 2 * len(self.poses)

    def pose_column(self, photo_id: str) -> int:
        photo_ids = list(self.poses.keys())
        return self.wall_count + 2 * photo_ids.index(photo_id)

    def neighbours(self, wall_index: int) -> list:
        before = (wall_index - 1) % self.wall_count
        after = (wall_index + 1) % self.wall_count
        return [before, after]

    def is_normal_corner(self, wall_index: int, neighbour_index: int) -> bool:
        before, after = self.neighbours(wall_index)
        if neighbour_index == after:
            first, second = self.directions[wall_index], self.directions[neighbour_index]
        else:
            first, second = self.directions[neighbour_index], self.directions[wall_index]
        return first[0] * second[1] - first[1] * second[0] < 0

    def build_rows(self):
        rows = []
        targets = []
        weights = []
        used = []

        for observation in self.observations:
            if not observation.active:
                continue

            if observation.kind == "distance":
                row = np.zeros(self.unknown_count())
                column = self.pose_column(observation.photo_id)
                normal = self.normals[observation.wall_index]
                row[column:column + 2] = normal
                row[observation.wall_index] = -1.0
                rows.append(row)
                targets.append(observation.distance)
                weights.append(1.0 / observation.sigma)
                used.append(observation)

        for observation in self.observations:
            if observation.kind != "inside_active":
                continue

            pose = self.poses[observation.photo_id]
            column = self.pose_column(observation.photo_id)
            rotated = pose.rotate(observation.point)
            normal = self.normals[observation.limit_wall]

            row = np.zeros(self.unknown_count())
            row[column:column + 2] = normal
            row[observation.limit_wall] = -1.0
            rows.append(row)
            targets.append(-float(normal @ rotated))
            weights.append(1.0 / BASE_SIGMA_M)
            used.append(observation)

        for wall_index, position in self.wall_priors.items():
            row = np.zeros(self.unknown_count())
            row[wall_index] = 1.0
            rows.append(row)
            targets.append(float(position))
            weights.append(1.0 / WALL_PRIOR_SIGMA_M)
            used.append(None)

        prior_count = 0
        for photo_id, pose in self.poses.items():
            if pose.prior is None:
                continue

            column = self.pose_column(photo_id)
            for axis in range(2):
                row = np.zeros(self.unknown_count())
                row[column + axis] = 1.0
                rows.append(row)
                targets.append(float(pose.prior[axis]))
                weights.append(1.0 / PRIOR_SIGMA_M)
                used.append(None)
            prior_count += 1

        if prior_count == 0:
            for row in self.gauge_rows():
                rows.append(row)
                targets.append(0.0)
                weights.append(100.0)
                used.append(None)

        return np.array(rows), np.array(targets), np.array(weights), used

    def gauge_rows(self) -> list:
        rows = []

        first = np.zeros(self.unknown_count())
        first[0] = 1.0
        rows.append(first)

        for wall_index in range(1, self.wall_count):
            if abs(self.normals[wall_index] @ self.normals[0]) < 0.5:
                second = np.zeros(self.unknown_count())
                second[wall_index] = 1.0
                rows.append(second)
                break

        return rows

    def solve_once(self):
        matrix, targets, weights, used = self.build_rows()
        weighted_matrix = matrix * weights[:, None]
        weighted_targets = targets * weights

        solution, _, rank, _ = np.linalg.lstsq(weighted_matrix, weighted_targets, rcond=None)
        if rank < self.unknown_count():
            self.notes.append("some walls are not tied to any photo, their position is a guess")

        residuals = matrix @ solution - targets
        return solution, residuals, used, weighted_matrix

    def apply_solution(self, solution: np.ndarray):
        self.wall_positions = solution[:self.wall_count]
        for photo_id in self.poses:
            column = self.pose_column(photo_id)
            self.poses[photo_id].position = solution[column:column + 2]

    def remove_outliers(self, residuals: np.ndarray, used: list) -> bool:
        removed = False

        for residual, observation in zip(residuals, used):
            if observation is None or observation.kind != "distance":
                continue

            limit = max(OUTLIER_LIMIT * observation.sigma, MIN_OUTLIER_M)
            if abs(residual) > limit:
                observation.active = False
                removed = True
                self.notes.append(f"{observation.photo_id}: distance to W{observation.wall_index + 1} "
                                  f"off by {residual:.2f} m, ignored")

        return removed

    def add_inside_limits(self) -> bool:
        added = False

        for observation in list(self.observations):
            if observation.kind != "inside" or observation.photo_id not in self.poses:
                continue

            pose = self.poses[observation.photo_id]
            room_point = pose.to_room(observation.point)

            for limit_wall in self.neighbours(observation.wall_index):
                if abs(self.normals[limit_wall] @ self.normals[observation.wall_index]) > 0.5:
                    continue
                if not self.is_normal_corner(observation.wall_index, limit_wall):
                    continue

                inside_distance = self.normals[limit_wall] @ room_point - self.wall_positions[limit_wall]
                if inside_distance < -0.02:
                    limit = Observation(observation.photo_id, observation.wall_index, 0.0, kind="inside_active")
                    limit.point = observation.point
                    limit.limit_wall = limit_wall
                    self.observations.append(limit)
                    added = True

            observation.kind = "inside_checked"

        return added

    def estimate_sigmas(self, weighted_matrix: np.ndarray, residuals: np.ndarray, used: list):
        distance_residuals = []
        for residual, observation in zip(residuals, used):
            if observation is not None and observation.kind == "distance":
                distance_residuals.append(residual / observation.sigma)

        scale = 1.0
        if len(distance_residuals) > self.unknown_count():
            scale = max(float(np.sqrt(np.mean(np.square(distance_residuals)))), 1.0)

        covariance = np.linalg.pinv(weighted_matrix.T @ weighted_matrix) * scale ** 2
        self.wall_sigmas = np.sqrt(np.clip(np.diag(covariance)[:self.wall_count], 0, None))

    def solve(self):
        if not self.poses:
            raise ValueError("No photo could be matched to the room walls")

        for round_number in range(3):
            solution, residuals, used, weighted_matrix = self.solve_once()
            self.apply_solution(solution)
            if not self.remove_outliers(residuals, used):
                break

        for round_number in range(INSIDE_ROUNDS):
            if not self.add_inside_limits():
                break
            solution, residuals, used, weighted_matrix = self.solve_once()
            self.apply_solution(solution)

        self.estimate_sigmas(weighted_matrix, residuals, used)

    def corner(self, wall_a: int, wall_b: int) -> np.ndarray:
        matrix = np.array([self.normals[wall_a], self.normals[wall_b]])
        targets = np.array([self.wall_positions[wall_a], self.wall_positions[wall_b]])
        return np.linalg.solve(matrix, targets)

    def wall_ends(self, wall_index: int) -> tuple:
        before, after = self.neighbours(wall_index)
        return self.corner(before, wall_index), self.corner(wall_index, after)

    def wall_length_sigma(self, wall_index: int) -> float:
        before, after = self.neighbours(wall_index)
        return float(math.hypot(self.wall_sigmas[before], self.wall_sigmas[after]))

    def position_along_wall(self, photo_id: str, wall_index: int, photo_point) -> float:
        if photo_id not in self.poses:
            return None

        start, end = self.wall_ends(wall_index)
        room_point = self.poses[photo_id].to_room(photo_point)
        return float(self.directions[wall_index] @ (room_point - start))
