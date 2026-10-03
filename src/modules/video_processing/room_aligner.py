import math

import numpy as np


MAX_ROTATION_SPREAD_DEG = 15


class ChunkTransform:

    def __init__(self, rotation: np.ndarray, translation: np.ndarray, scale: float):
        self.rotation = rotation
        self.translation = translation
        self.scale = scale

    def apply_to_pose(self, camera_to_world: np.ndarray) -> np.ndarray:
        moved = np.eye(4)
        moved[:3, :3] = self.rotation @ camera_to_world[:3, :3]
        moved[:3, 3] = self.scale * self.rotation @ camera_to_world[:3, 3] + self.translation
        return moved

    def then(self, outer):
        rotation = outer.rotation @ self.rotation
        translation = outer.scale * outer.rotation @ self.translation + outer.translation
        return ChunkTransform(rotation, translation, outer.scale * self.scale)


class ChunkAligner:

    def identity(self) -> ChunkTransform:
        return ChunkTransform(np.eye(3), np.zeros(3), 1.0)

    def nearest_rotation(self, matrix: np.ndarray) -> np.ndarray:
        left, values, right = np.linalg.svd(matrix)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        return rotation

    def depth_ratio(self, pose_reference: dict, pose_other: dict):
        reference = pose_reference["depth"][pose_reference["mask"]]
        other = pose_other["depth"][pose_other["mask"]]
        reference = reference[np.isfinite(reference) & (reference > 0)]
        other = other[np.isfinite(other) & (other > 0)]

        if len(reference) == 0 or len(other) == 0:
            return None
        return float(np.median(reference) / np.median(other))

    def rotation_angle_deg(self, rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
        cosine = (np.trace(rotation_a.T @ rotation_b) - 1) / 2
        return math.degrees(math.acos(float(np.clip(cosine, -1, 1))))

    def align(self, shared_pairs: list) -> dict:
        if not shared_pairs:
            return {"transform": None, "note": "no shared doorway frames"}

        rotations = []
        ratios = []
        for pose_reference, pose_other in shared_pairs:
            rotations.append(pose_reference["camera_to_world"][:3, :3] @ pose_other["camera_to_world"][:3, :3].T)
            ratio = self.depth_ratio(pose_reference, pose_other)
            if ratio is not None:
                ratios.append(ratio)

        rotation = self.nearest_rotation(np.sum(np.array(rotations), axis=0))
        scale = float(np.median(ratios)) if ratios else 1.0

        spread = max(self.rotation_angle_deg(rotation, item) for item in rotations)
        if spread > MAX_ROTATION_SPREAD_DEG:
            return {"transform": None, "note": f"shared doorway frames disagree by {spread:.0f} deg"}

        translations = []
        for pose_reference, pose_other in shared_pairs:
            centre_reference = pose_reference["camera_to_world"][:3, 3]
            centre_other = pose_other["camera_to_world"][:3, 3]
            translations.append(centre_reference - scale * rotation @ centre_other)

        translation = np.median(np.array(translations), axis=0)
        note = f"aligned with {len(shared_pairs)} shared frames (rotation spread {spread:.1f} deg, scale {scale:.3f})"

        return {"transform": ChunkTransform(rotation, translation, scale), "note": note}
