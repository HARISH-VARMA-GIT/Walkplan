import os

from langchain_core.messages import HumanMessage, SystemMessage

from models.image_models import RoomImageAnalysis
from utils.image_utils import ImageConverter
from utils.llm_provider import LlmClient, VisionLimiter
from utils.prompts import ROOM_IMAGES_PROMPT, ROOM_IMAGES_SYSTEM_PROMPT


class ImageProcessingService:

    def __init__(self):
        self.image_converter = ImageConverter()
        self.llm_client = LlmClient(
            model_env_name="LLM_VISION_MODEL_CHAIN",
            default_models="gpt-4o-mini,gpt-4o",
            timeout=60,
            max_tokens=2048,
        )
        self.vision_limiter = VisionLimiter()

    def process_image(self, folder_path: str) -> dict:
        image_paths = self.image_converter.list_image_paths(folder_path)

        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")

        file_names = [os.path.basename(image_path) for image_path in image_paths]
        messages = self.build_messages(image_paths, file_names)

        self.vision_limiter.acquire()

        try:
            analysis = self.llm_client.run("gpt-4o", messages, RoomImageAnalysis)
        finally:
            self.vision_limiter.release()

        return {
            "folder_path": folder_path,
            "image_count": len(image_paths),
            "analysis": analysis.model_dump(),
        }

    def build_messages(self, image_paths: list[str], file_names: list[str]) -> list:
        prompt = ROOM_IMAGES_PROMPT.format(file_names="\n".join(file_names))
        content = [{"type": "text", "text": prompt}]

        for image_path in image_paths:
            base64_image = self.image_converter.convert_to_base64_jpeg(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        return [SystemMessage(content=ROOM_IMAGES_SYSTEM_PROMPT), HumanMessage(content=content)]
