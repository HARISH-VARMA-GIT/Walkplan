import math

import numpy as np

from models.photo_geometry import PhotoPlacement, RoomPhotoGeometry, RoomPlacements
from modules.image_processing.photo_measurer import FloorFrame


MIN_WALL_LENGTH_M = 0.3
UP_AGREEMENT_DEG = 15
MIN_CAMERA_HEIGHT_M = 0.5
MAX_CAMERA_HEIGHT_M = 2.2


class PhotoPlacer:

    def world_up(self, photo_set: RoomPhotoGeometry, poses: list) -> np.ndarray:
        ups = []

        for photo, pose in zip(photo_set.photos, poses):
            if photo.floor_found:
                rotation = pose["camera_to_world"][:3, :3]
                ups.append(rotation @ np.array(photo.up_vector))

        if not ups:
            rotation = poses[0]["camera_to_world"][:3, :3]
            ups.append(rotation @ np.array([0.0, -1.0, 0.0]))

        up = np.median(np.array(ups), axis=0)
        return up / np.linalg.norm(up)

    def up_from_cameras(self, poses: list, floor_normals: list) -> np.ndarray:
        camera_ups = []
        for pose in poses:
            camera_ups.append(pose["camera_to_world"][:3, :3] @ np.array([0.0, -1.0, 0.0]))

        up = np.mean(np.array(camera_ups), axis=0)
        up = up / np.linalg.norm(up)

        agreeing = []
        for pose, normal in zip(poses, floor_normals):
            if normal is None:
                continue
            world_normal = pose["camera_to_world"][:3, :3] @ np.array(normal)
            if world_normal @ up > math.cos(math.radians(UP_AGREEMENT_DEG)):
                agreeing.append(world_normal)

        if len(agreeing) >= 2:
            up = np.median(np.array(agreeing), axis=0)
            up = up / np.linalg.norm(up)

        return up

    def up_hints(self, poses: list, up: np.ndarray) -> list:
        hints = []
        for pose in poses:
            hints.append(pose["camera_to_world"][:3, :3].T @ up)
        return hints

    def shared_scale(self, moge_depths: list, poses: list) -> float:
        scales = []
        for moge_depth, pose in zip(moge_depths, poses):
            scale = self.depth_scale(moge_depth, pose)
            if scale is not None:
                scales.append(scale)

        return float(np.median(scales)) if scales else 1.0

    def floor_hints(self, poses: list, up: np.ndarray, scale: float, floor_offsets: list) -> list:
        camera_heights = []
        for pose in poses:
            camera_heights.append(float(pose["camera_to_world"][:3, 3] @ up) * scale)

        floor_levels = []
        for camera_height, floor_offset in zip(camera_heights, floor_offsets):
            if floor_offset is not None and MIN_CAMERA_HEIGHT_M < floor_offset < MAX_CAMERA_HEIGHT_M:
                floor_levels.append(camera_height - floor_offset)

        if not floor_levels:
            return [None] * len(poses)

        floor_level = float(np.median(floor_levels))
        return [camera_height - floor_level for camera_height in camera_heights]

    def world_axes(self, up: np.ndarray) -> tuple:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(helper @ up) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])

        axis_y = helper - (helper @ up) * up
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_x = np.cross(axis_y, up)

        return axis_x, axis_y

    def depth_scale(self, moge_depth: dict, pose: dict) -> float:
        moge_z = moge_depth["points"][..., 2][moge_depth["mask"]]
        moge_z = moge_z[np.isfinite(moge_z) & (moge_z > 0)]

        pose_z = pose["depth"][pose["mask"]]
        pose_z = pose_z[np.isfinite(pose_z) & (pose_z > 0)]

        if len(moge_z) == 0 or len(pose_z) == 0:
            return None

        return float(np.median(moge_z) / np.median(pose_z))

    def heading_and_position(self, photo, pose: dict, axis_x: np.ndarray, axis_y: np.ndarray, scale: float) -> tuple:
        frame = FloorFrame(np.array(photo.up_vector))
        rotation = pose["camera_to_world"][:3, :3]
        translation = pose["camera_to_world"][:3, 3]

        right_world = rotation @ frame.right
        heading = math.atan2(right_world @ axis_y, right_world @ axis_x)
        position = np.array([translation @ axis_x, translation @ axis_y]) * scale

        return heading, position

    def wall_angle(self, photo_set: RoomPhotoGeometry, headings: dict) -> float:
        sum_cos = 0.0
        sum_sin = 0.0

        for photo in photo_set.photos:
            for wall in photo.walls:
                if wall.visible_length_m < MIN_WALL_LENGTH_M:
                    continue

                direction = np.array(wall.end) - np.array(wall.start)
                angle = math.atan2(direction[1], direction[0]) + headings[photo.photo_id]
                sum_cos += wall.visible_length_m * math.cos(4 * angle)
                sum_sin += wall.visible_length_m * math.sin(4 * angle)

        return math.atan2(sum_sin, sum_cos) / 4

    def rotate(self, point: np.ndarray, angle: float) -> np.ndarray:
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        return np.array([cos_value * point[0] - sin_value * point[1], sin_value * point[0] + cos_value * point[1]])

    def place(self, photo_set: RoomPhotoGeometry, poses: list, moge_depths: list) -> RoomPlacements:
        up = self.world_up(photo_set, poses)
        axis_x, axis_y = self.world_axes(up)

        photo_scales = []
        for moge_depth, pose in zip(moge_depths, poses):
            photo_scales.append(self.depth_scale(moge_depth, pose))

        valid_scales = [value for value in photo_scales if value is not None]
        scale = float(np.median(valid_scales)) if valid_scales else 1.0

        headings = {}
        positions = {}
        for photo, pose in zip(photo_set.photos, poses):
            heading, position = self.heading_and_position(photo, pose, axis_x, axis_y, scale)
            headings[photo.photo_id] = heading
            positions[photo.photo_id] = position

        wall_angle = self.wall_angle(photo_set, headings)

        placements = []
        for photo, photo_scale in zip(photo_set.photos, photo_scales):
            heading = headings[photo.photo_id] - wall_angle
            position = self.rotate(positions[photo.photo_id], -wall_angle)

            placements.append(PhotoPlacement(
                photo_id=photo.photo_id,
                heading_deg=round(math.degrees(heading), 2),
                position=(round(float(position[0]), 3), round(float(position[1]), 3)),
                depth_scale=round(photo_scale, 4) if photo_scale is not None else scale,
            ))

        return RoomPlacements(scale=round(scale, 4), wall_angle_deg=round(math.degrees(wall_angle), 2), placements=placements)
