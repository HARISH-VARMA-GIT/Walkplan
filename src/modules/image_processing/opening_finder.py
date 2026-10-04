import numpy as np
from PIL import Image

from models.photo_geometry import PhotoOpening
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamProcessor, SamModel

DETECTOR_NAME = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_NAME = "facebook/sam-vit-base"
MIRROR_LABEL = "mirror"
OPENING_WORDS = ["door", "window", MIRROR_LABEL]
DUPLICATE_OVERLAP = 0.5
MAX_BOX_SHARE = 0.4
MAX_WALL_DISTANCE_M = 0.35

VALID_SIZES = {
    "door": {"width": (0.6, 1.3), "height": (1.8, 2.6)},
    "window": {"width": (0.3, 3.0), "height": (0.3, 2.8)},
}


class ObjectDetector:

    def __init__(self, model_name: str = DETECTOR_NAME):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device).eval()

    def find_boxes(self, image_rgb: np.ndarray, words: list, box_threshold=0.3, text_threshold=0.25) -> list:
        height, width = image_rgb.shape[:2]
        text = ". ".join(words) + "."

        inputs = self.processor(images=Image.fromarray(image_rgb), text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )[0]

        labels = results.get("text_labels", results.get("labels"))
        detections = []

        for index in range(len(results["scores"])):
            label = self.clean_label(str(labels[index]), words)
            if label is None:
                continue

            detections.append({
                "label": label,
                "box": [float(value) for value in results["boxes"][index].cpu().numpy()],
                "score": float(results["scores"][index]),
            })

        return self.remove_duplicates(detections)

    def clean_label(self, label: str, words: list):
        for word in words:
            if word in label:
                return word
        return None

    def overlap(self, box_a: list, box_b: list) -> float:
        left = max(box_a[0], box_b[0])
        top = max(box_a[1], box_b[1])
        right = min(box_a[2], box_b[2])
        bottom = min(box_a[3], box_b[3])

        shared = max(0.0, right - left) * max(0.0, bottom - top)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return shared / max(area_a + area_b - shared, 1e-6)

    def score_of(self, detection: dict) -> float:
        return detection["score"]

    def remove_duplicates(self, detections: list) -> list:
        kept = []

        for detection in sorted(detections, key=self.score_of, reverse=True):
            is_duplicate = False

            for other in kept:
                if self.overlap(detection["box"], other["box"]) > DUPLICATE_OVERLAP:
                    is_duplicate = True

            if not is_duplicate:
                kept.append(detection)

        return kept


class ObjectSegmenter:

    def __init__(self, model_name: str = SEGMENTER_NAME):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device).eval()

    def segment(self, image_rgb: np.ndarray, box: list) -> np.ndarray:

        inputs = self.processor(Image.fromarray(image_rgb), input_boxes=[[box]], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        return masks[0][0][0].numpy().astype(bool)


class OpeningFinder:

    def __init__(self):
        self.detector = None
        self.segmenter = None

    def load_models(self):
        if self.detector is None:
            self.detector = ObjectDetector()
            self.segmenter = ObjectSegmenter()

    def unload_models(self):
        if self.detector is None:
            return

        import torch

        self.detector = None
        self.segmenter = None
        torch.cuda.empty_cache()

    def find_openings(self, image_rgb: np.ndarray) -> list:
        self.load_models()
        found = self.detector.find_boxes(image_rgb, OPENING_WORDS)
        image_area = image_rgb.shape[0] * image_rgb.shape[1]

        detections = []
        for detection in found:
            box = detection["box"]
            if (box[2] - box[0]) * (box[3] - box[1]) <= MAX_BOX_SHARE * image_area:
                detections.append(detection)

        for detection in detections:
            detection["mask"] = self.segmenter.segment(image_rgb, detection["box"])

        return detections

    def openings_only(self, detections: list) -> list:
        openings = []

        for detection in detections:
            if detection["label"] != MIRROR_LABEL:
                openings.append(detection)

        return openings

    def combined_mask(self, detections: list, shape: tuple) -> np.ndarray:
        combined = np.zeros(shape, dtype=bool)

        for detection in detections:
            combined = combined | detection["mask"]

        return combined


class OpeningMeasurer:

    def measure(self, analysis, detections: list, depth: dict) -> list:
        openings = []

        for index, detection in enumerate(detections):
            opening_id = f"{analysis.geometry.photo_id}-O{index + 1}"
            opening = PhotoOpening(id=opening_id, type=detection["label"], box=detection["box"], score=detection["score"])

            wall_line = self.find_wall(analysis, detection, depth)
            if wall_line is not None:
                self.measure_on_wall(opening, wall_line, analysis, detection["mask"])

            openings.append(opening)

        return openings

    def find_wall(self, analysis, detection: dict, depth: dict):
        mask = detection["mask"] & depth["mask"]
        points = depth["points"][mask]
        points = points[np.isfinite(points).all(axis=1)]

        if len(points) < 20:
            return None

        box = detection["box"]
        box_centre = (box[0] + box[2]) / 2 / analysis.geometry.image_width

        best_wall = None
        best_distance = None

        for wall_line in analysis.wall_lines:
            cols = wall_line.plane.cols / analysis.geometry.image_width
            left = np.percentile(cols, 2) - 0.05
            right = np.percentile(cols, 98) + 0.05

            if not left <= box_centre <= right:
                continue

            distance = float(np.median(np.abs(wall_line.plane.distance(points))))
            if best_distance is None or distance < best_distance:
                best_wall = wall_line
                best_distance = distance

        if best_distance is None or best_distance > MAX_WALL_DISTANCE_M:
            return None

        return best_wall

    def pixel_rays(self, rows: np.ndarray, cols: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
        x = ((cols + 0.5) / width - intrinsics[0, 2]) / intrinsics[0, 0]
        y = ((rows + 0.5) / height - intrinsics[1, 2]) / intrinsics[1, 1]
        z = np.ones_like(x)
        return np.stack([x, y, z], axis=-1)

    def measure_on_wall(self, opening: PhotoOpening, wall_line, analysis, mask: np.ndarray):
        rows, cols = np.nonzero(mask)
        if len(rows) > 20000:
            pick = np.linspace(0, len(rows) - 1, 20000).astype(int)
            rows, cols = rows[pick], cols[pick]

        width = analysis.geometry.image_width
        height = analysis.geometry.image_height
        rays = self.pixel_rays(rows, cols, analysis.intrinsics, width, height)

        plane = wall_line.plane
        facing = rays @ plane.normal
        usable = facing < -1e-6
        if usable.sum() < 20:
            return

        distances = -plane.offset / facing[usable]
        points = rays[usable] * distances[:, None]

        frame = analysis.frame
        positions = frame.to_2d(points) @ wall_line.direction
        heights = frame.height(points)

        near_edge = float(np.percentile(positions, 2))
        far_edge = float(np.percentile(positions, 98))
        bottom = float(np.percentile(heights, 2))
        top = float(np.percentile(heights, 98))

        opening.wall_letter = wall_line.letter
        opening.start_along_wall_m = round(near_edge - wall_line.start_s, 3)
        opening.width_m = round(far_edge - near_edge, 3)

        if opening.type == "door":
            opening.height_m = round(top, 3)
        else:
            opening.height_m = round(top - bottom, 3)
            opening.sill_height_m = round(bottom, 3)

        opening.looks_valid = self.is_valid(opening, analysis)

    def is_valid(self, opening: PhotoOpening, analysis) -> bool:
        box = opening.box
        image_width = analysis.geometry.image_width
        image_height = analysis.geometry.image_height

        cut_left_or_right = box[0] < 3 or box[2] > image_width - 3
        cut_at_top = box[1] < 3 and opening.type == "door"

        if cut_left_or_right or cut_at_top:
            return False

        limits = VALID_SIZES[opening.type]
        width_ok = limits["width"][0] <= opening.width_m <= limits["width"][1]

        if not analysis.geometry.floor_found:
            return width_ok

        height_ok = limits["height"][0] <= opening.height_m <= limits["height"][1]
        return width_ok and height_ok
