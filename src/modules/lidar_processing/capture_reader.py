import csv
import os

import cv2
import numpy as np




class CameraPose:

    def __init__(self, frame_index: int, time_seconds: float, rotation: np.ndarray, position: np.ndarray):
        self.frame_index = frame_index
        self.time_seconds = time_seconds
        self.rotation = rotation
        self.position = position

    def to_world(self, camera_points: np.ndarray) -> np.ndarray:
        return camera_points @ self.rotation.T + self.position

    def direction_to_world(self, camera_directions: np.ndarray) -> np.ndarray:
        return camera_directions @ self.rotation.T

    def to_camera(self, world_points: np.ndarray) -> np.ndarray:
        return (world_points - self.position) @ self.rotation

    def direction_to_camera(self, world_direction: np.ndarray) -> np.ndarray:
        return world_direction @ self.rotation


class StrayCapture:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.name = os.path.basename(os.path.normpath(folder_path))
        self.depth_folder = os.path.join(folder_path, "depth")
        self.confidence_folder = os.path.join(folder_path, "confidence")
        self.video_path = os.path.join(folder_path, "rgb.mp4")

        self.camera_matrix = np.loadtxt(os.path.join(folder_path, "camera_matrix.csv"), delimiter=",")
        self.poses = self.read_poses(os.path.join(folder_path, "odometry.csv"))
        self.depth_files = self.read_depth_files()

        first_depth = self.read_depth(self.depth_files[0])
        self.depth_height, self.depth_width = first_depth.shape
        self.image_width, self.image_height = self.read_image_size()

    def quaternion_to_rotation(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])

    def read_poses(self, odometry_path: str) -> dict:
        poses = {}

        with open(odometry_path, "r") as odometry_file:
            rows = csv.reader(odometry_file)
            next(rows)

            for row in rows:
                values = [float(value.strip()) for value in row[:9]]
                frame_index = int(values[1])
                rotation = self.quaternion_to_rotation(values[5], values[6], values[7], values[8])
                position = np.array(values[2:5])
                poses[frame_index] = CameraPose(frame_index, values[0], rotation, position)

        return poses

    def read_depth_files(self) -> dict:
        depth_files = {}

        for file_name in sorted(os.listdir(self.depth_folder)):
            if file_name.endswith(".png"):
                frame_index = int(os.path.splitext(file_name)[0])
                depth_files[frame_index] = file_name

        return depth_files

    def read_image_size(self) -> tuple:
        if os.path.exists(self.video_path):
            video = cv2.VideoCapture(self.video_path)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video.release()
            if width > 0 and height > 0:
                return width, height

        return int(round(self.camera_matrix[0, 2] * 2)), int(round(self.camera_matrix[1, 2] * 2))

    def read_depth(self, file_name: str) -> np.ndarray:
        depth = cv2.imread(os.path.join(self.depth_folder, file_name), cv2.IMREAD_UNCHANGED)
        return depth.astype(np.float32) / 1000.0

    def read_confidence(self, file_name: str):
        path = os.path.join(self.confidence_folder, file_name)
        if not os.path.exists(path):
            return None
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def depth_intrinsics(self) -> np.ndarray:
        scale_x = self.depth_width / self.image_width
        scale_y = self.depth_height / self.image_height
        intrinsics = self.camera_matrix.copy()
        intrinsics[0] = intrinsics[0] * scale_x
        intrinsics[1] = intrinsics[1] * scale_y
        return intrinsics

    def frame_indexes(self) -> list:
        indexes = []
        for frame_index in self.depth_files:
            if frame_index in self.poses:
                indexes.append(frame_index)
        return indexes

    def camera_path(self) -> np.ndarray:
        positions = []
        for frame_index in sorted(self.poses):
            positions.append(self.poses[frame_index].position)
        return np.array(positions)
