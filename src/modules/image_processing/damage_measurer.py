import logging
import os

import cv2
import numpy as np

from models.photo_geometry import PhotoDamage
from modules.image_processing.damage_box_picker import DamageBoxPicker
from modules.image_processing.opening_finder import OpeningFinder, OpeningMeasurer


logger = logging.getLogger(__name__)

GENERIC_DAMAGE_WORDS = ["stain", "crack", "spot", "mark", "hole", "peeling paint"]
CANDIDATE_THRESHOLD = 0.15
MAX_CANDIDATES = 8
MAX_DAMAGE_BOX_SHARE = 0.4
FALLBACK_BOX_SHARE = 0.3
MAX_WALL_DISTANCE_M = 0.25
HORIZONTAL_LIMIT = 0.8
MIN_POINTS = 20
FLOOR_LIMIT_M = 0.5
DAMAGE_BOX_COLOR = (255, 60, 0)


class DamageMeasurer:

    def __init__(self, opening_finder: OpeningFinder):
        self.opening_finder = opening_finder
        self.opening_measurer = OpeningMeasurer()
        self.box_picker = DamageBoxPicker() if os.getenv("OPENAI_API_KEY") else None

    def detector_words(self, search_terms: list) -> list:
        words = []
        for term in search_terms:
            for word in [term, term.split()[-1]]:
                if word not in words:
                    words.append(word)
        return words

    def centre_distance(self, box: list, width: int, height: int) -> float:
        centre_x = (box[0] + box[2]) / 2 / width - 0.5
        centre_y = (box[1] + box[3]) / 2 / height - 0.5
        return float(np.hypot(centre_x, centre_y))

    def find_candidates(self, image_rgb: np.ndarray, search_terms: list) -> list:
        self.opening_finder.load_models()
        height, width = image_rgb.shape[:2]

        words = self.detector_words(search_terms + GENERIC_DAMAGE_WORDS)
        found = self.opening_finder.detector.find_boxes(image_rgb, words, box_threshold=CANDIDATE_THRESHOLD, text_threshold=CANDIDATE_THRESHOLD)

        candidates = []
        for detection in found:
            box = detection["box"]
            if (box[2] - box[0]) * (box[3] - box[1]) > MAX_DAMAGE_BOX_SHARE * width * height:
                continue
            detection["rank"] = detection["score"] * (1 - self.centre_distance(box, width, height))
            candidates.append(detection)

        candidates.sort(key=self.rank_of, reverse=True)
        return candidates[:MAX_CANDIDATES]

    def rank_of(self, detection: dict) -> float:
        return detection["rank"]

    def choose_box(self, image_rgb: np.ndarray, request: dict) -> dict:
        height, width = image_rgb.shape[:2]
        candidates = self.find_candidates(image_rgb, request["search_terms"])

        if self.box_picker is None:
            if candidates:
                return {"box": candidates[0]["box"], "score": candidates[0]["score"], "method": "measured", "use_mask": True}
            return {"box": self.fallback_box(width, height), "score": 0.0, "method": "assumed", "use_mask": False}

        boxes = [candidate["box"] for candidate in candidates]
        picked = self.box_picker.pick(image_rgb, boxes, request)["choice"]
        logger.info("%s: picked box %d of %d, %s (%s)", request["damage_id"], picked["box_number"], len(boxes), picked["grid_cell"], picked["reason"])

        if 1 <= picked["box_number"] <= len(candidates):
            candidate = candidates[picked["box_number"] - 1]
            return {"box": candidate["box"], "score": candidate["score"], "method": "measured", "use_mask": True}

        if picked["grid_cell"] == "not visible":
            return None

        grid_box = self.box_picker.grid_box(picked["grid_cell"], width, height)
        if grid_box is not None:
            return {"box": grid_box, "score": 0.0, "method": "estimated", "use_mask": False}

        return {"box": self.fallback_box(width, height), "score": 0.0, "method": "assumed", "use_mask": False}

    def fallback_box(self, width: int, height: int) -> list:
        side = np.sqrt(FALLBACK_BOX_SHARE)
        return [width * (1 - side) / 2, height * (1 - side) / 2, width * (1 + side) / 2, height * (1 + side) / 2]

    def box_mask(self, box: list, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        left, top, right, bottom = [int(round(value)) for value in box]
        mask[max(top, 0):bottom, max(left, 0):right] = True
        return mask

    def surface_normal(self, points: np.ndarray) -> np.ndarray:
        centred = points - points.mean(axis=0)
        return np.linalg.svd(centred, full_matrices=False)[2][-1]

    def closest_wall(self, analysis, points: np.ndarray):
        best_wall = None
        best_distance = None

        for wall_line in analysis.wall_lines:
            distance = float(np.median(np.abs(wall_line.plane.distance(points))))
            if best_distance is None or distance < best_distance:
                best_wall = wall_line
                best_distance = distance

        if best_distance is None or best_distance > MAX_WALL_DISTANCE_M:
            return None
        return best_wall

    def points_on_wall(self, analysis, wall_line, mask: np.ndarray):
        rows, cols = np.nonzero(mask)
        if len(rows) > 20000:
            pick = np.linspace(0, len(rows) - 1, 20000).astype(int)
            rows, cols = rows[pick], cols[pick]

        rays = self.opening_measurer.pixel_rays(rows, cols, analysis.intrinsics, analysis.geometry.image_width, analysis.geometry.image_height)
        facing = rays @ wall_line.plane.normal
        usable = facing < -1e-6
        if usable.sum() < MIN_POINTS:
            return None

        distances = -wall_line.plane.offset / facing[usable]
        return rays[usable] * distances[:, None]

    def choose_surface(self, analysis, points: np.ndarray, surface_hint: str) -> tuple:
        frame = analysis.frame
        normal = self.surface_normal(points)

        if abs(normal @ frame.up) > HORIZONTAL_LIMIT:
            centre_height = float(np.median(frame.height(points)))
            if frame.floor_offset is not None:
                return ("floor" if centre_height < FLOOR_LIMIT_M else "ceiling"), None
            return ("floor" if centre_height < 0 else "ceiling"), None

        wall_line = self.closest_wall(analysis, points)
        if wall_line is not None:
            return "wall", wall_line

        if surface_hint in ["floor", "ceiling"]:
            return surface_hint, None
        return "wall", None

    def measure_one(self, analysis, image_rgb: np.ndarray, depth: dict, request: dict):
        height, width = image_rgb.shape[:2]
        chosen = self.choose_box(image_rgb, request)
        if chosen is None:
            return None

        box = chosen["box"]
        mask = self.box_mask(box, (height, width))
        if chosen["use_mask"]:
            segmented = self.opening_finder.segmenter.segment(image_rgb, box) & mask
            if segmented.sum() >= MIN_POINTS:
                mask = segmented

        damage = PhotoDamage(damage_id=request["damage_id"], surface_type="wall", box=[round(value, 1) for value in box],
                             score=round(chosen["score"], 3), method=chosen["method"])

        usable = mask & depth["mask"]
        points = depth["points"][usable]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < MIN_POINTS:
            if request.get("surface_hint") in ["floor", "ceiling"]:
                damage.surface_type = request["surface_hint"]
            return damage

        surface_type, wall_line = self.choose_surface(analysis, points, request.get("surface_hint", "unknown"))
        damage.surface_type = surface_type

        if wall_line is not None:
            on_wall = self.points_on_wall(analysis, wall_line, mask)
            if on_wall is not None:
                points = on_wall

        self.fill_measurements(damage, analysis, points, wall_line, mask, box)
        return damage

    def fill_measurements(self, damage: PhotoDamage, analysis, points: np.ndarray, wall_line, mask: np.ndarray, box: list):
        frame = analysis.frame
        points_2d = frame.to_2d(points)
        heights = frame.height(points)

        box_pixels = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
        fill = min(float(mask.sum()) / box_pixels, 1.0)

        if wall_line is not None:
            positions = points_2d @ wall_line.direction
            near_edge = float(np.percentile(positions, 2))
            far_edge = float(np.percentile(positions, 98))
            centre = (near_edge + far_edge) / 2
            span = float(np.percentile(heights, 98) - np.percentile(heights, 2))

            damage.wall_letter = wall_line.letter
            damage.start_along_wall_m = round(near_edge - wall_line.start_s, 3)
            damage.width_m = round(far_edge - near_edge, 3)
            damage.area_m2 = round((far_edge - near_edge) * span * fill, 3)
            point = wall_line.point_at(centre)
        else:
            low = np.percentile(points_2d, 2, axis=0)
            high = np.percentile(points_2d, 98, axis=0)
            size = high - low
            damage.width_m = round(float(max(size)), 3)
            damage.area_m2 = round(float(size[0] * size[1]) * fill, 3)
            point = np.median(points_2d, axis=0)

        damage.point = (round(float(point[0]), 3), round(float(point[1]), 3))

        if frame.floor_offset is not None:
            damage.center_height_m = round(float(np.median(heights)), 3)

    def measure(self, analysis, image_rgb: np.ndarray, depth: dict, requests: list) -> list:
        damages = []
        for request in requests:
            damage = self.measure_one(analysis, image_rgb, depth, request)
            if damage is None:
                logger.info("%s: not visible in %s", request["damage_id"], analysis.geometry.photo_id)
            else:
                damages.append(damage)
        return damages

    def save_still(self, image_rgb: np.ndarray, damage: PhotoDamage, photo_id: str, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        image = image_rgb.copy()
        left, top, right, bottom = [int(round(value)) for value in damage.box]
        thickness = max(image.shape[1] // 400, 2)

        cv2.rectangle(image, (left, top), (right, bottom), DAMAGE_BOX_COLOR, thickness)
        label = f"{damage.damage_id} {damage.surface_type}"
        if damage.method != "measured":
            label += f" ({damage.method})"
        cv2.putText(image, label, (left + 4, max(top - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, image.shape[1] / 1200, DAMAGE_BOX_COLOR, thickness, cv2.LINE_AA)

        path = os.path.join(folder, f"{damage.damage_id}_{photo_id}.jpg")
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return path
