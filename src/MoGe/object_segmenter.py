import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor


DETECTOR_NAME = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_NAME = "facebook/sam-vit-base"


class ObjectDetector:
    def __init__(self, model_name=DETECTOR_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device).eval()

    def find_box(self, image_rgb, text, box_threshold=0.3, text_threshold=0.25):
        pil_image = Image.fromarray(image_rgb)
        height, width = image_rgb.shape[:2]

        inputs = self.processor(images=pil_image, text=f"{text}.", return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )[0]

        if len(results["scores"]) == 0:
            return None

        best_index = int(torch.argmax(results["scores"]))
        box = results["boxes"][best_index].cpu().numpy()
        score = float(results["scores"][best_index])

        return box, score


class ObjectSegmenter:
    def __init__(self, model_name=SEGMENTER_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device).eval()

    def segment(self, image_rgb, box):
        pil_image = Image.fromarray(image_rgb)
        box_list = [[[float(value) for value in box]]]

        inputs = self.processor(pil_image, input_boxes=box_list, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        return masks[0][0][0].numpy().astype(bool)


class ObjectFinder:
    def __init__(self):
        self.detector = ObjectDetector()
        self.segmenter = ObjectSegmenter()

    def find_object_mask(self, image_rgb, text):
        found = self.detector.find_box(image_rgb, text)
        if found is None:
            return None

        box, score = found
        mask = self.segmenter.segment(image_rgb, box)

        return {"box": box, "score": score, "mask": mask}
