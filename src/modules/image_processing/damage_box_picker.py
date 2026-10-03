import base64
import os

import cv2
import numpy as np
from langchain_core.messages import HumanMessage

from models.video_models import DamageBoxChoice
from utils.llm_provider import LlmClient
from utils.prompts import DAMAGE_BOX_PROMPT


BOX_COLOR = (255, 0, 0)
GRID_CELLS = {
    "top-left": (0, 0), "top-center": (1, 0), "top-right": (2, 0),
    "middle-left": (0, 1), "center": (1, 1), "middle-right": (2, 1),
    "bottom-left": (0, 2), "bottom-center": (1, 2), "bottom-right": (2, 2),
}


class DamageBoxPicker:

    def __init__(self):
        self.llm_client = LlmClient(
            model_env_name="LLM_DAMAGE_VISION_MODEL_CHAIN",
            default_models="gpt-4.1,gpt-4o",
            timeout=60,
        )
        self.model_name = os.getenv("DAMAGE_VISION_MODEL_NAME")

    def draw_candidates(self, image_rgb: np.ndarray, boxes: list) -> str:
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        thickness = max(image.shape[1] // 500, 2)
        scale = image.shape[1] / 900

        for number, box in enumerate(boxes, start=1):
            left, top, right, bottom = [int(round(value)) for value in box]
            cv2.rectangle(image, (left, top), (right, bottom), BOX_COLOR[::-1], thickness)
            label_position = (left + 4, max(top + int(28 * scale), 20))
            cv2.putText(image, str(number), label_position, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
            cv2.putText(image, str(number), label_position, cv2.FONT_HERSHEY_SIMPLEX, scale, BOX_COLOR[::-1], thickness, cv2.LINE_AA)

        image_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def pick(self, image_rgb: np.ndarray, boxes: list, request: dict) -> dict:
        prompt = DAMAGE_BOX_PROMPT.format(
            damage_class=request.get("damage_class", "other").replace("_", " "),
            description=request.get("description", ""),
            quote=request.get("quote", ""),
        )

        image_base64 = self.draw_candidates(image_rgb, boxes)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        ]

        choice = self.llm_client.run(self.model_name, [HumanMessage(content=content)], DamageBoxChoice)
        return {"damage_id": request["damage_id"], "choice": choice.model_dump()}

    def grid_box(self, grid_cell: str, width: int, height: int):
        if grid_cell not in GRID_CELLS:
            return None

        column, row = GRID_CELLS[grid_cell]
        return [column * width / 3, row * height / 3, (column + 1) * width / 3, (row + 1) * height / 3]
