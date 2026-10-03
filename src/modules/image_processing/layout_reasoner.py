import os

from langchain_core.messages import HumanMessage, SystemMessage

from models.image_models import RoomLayout
from models.photo_geometry import PhotoGeometry
from utils.image_utils import ImageConverter
from utils.llm_provider import LlmClient, VisionLimiter
from utils.prompts import ROOM_LAYOUT_PROMPT, ROOM_LAYOUT_SYSTEM_PROMPT


class LayoutReasoner:

    def __init__(self):
        self.image_converter = ImageConverter()
        self.llm_client = LlmClient(
            model_env_name="LLM_LAYOUT_MODEL_CHAIN",
            default_models="gpt-4.1,gpt-4o",
            timeout=600,
            max_tokens=32000,
        )
        self.vision_limiter = VisionLimiter()
        self.layout_model_name = os.getenv("LAYOUT_MODEL_NAME")

    def describe_wall(self, wall) -> str:
        return (f"{wall.letter} visible {wall.visible_length_m:.2f} m "
                f"(left end {wall.start_type}, right end {wall.end_type}, {wall.distance_from_camera_m:.1f} m away)")

    def describe_photo(self, photo: PhotoGeometry) -> str:
        lines = [f"{photo.photo_id}:"]

        if photo.walls:
            wall_texts = [self.describe_wall(wall) for wall in photo.walls]
            lines.append("  walls: " + "; ".join(wall_texts))
        else:
            lines.append("  walls: none found")

        for corner in photo.corners:
            lines.append(f"  corner {corner.number}: between {corner.wall_before} and {corner.wall_after}, "
                         f"{corner.corner_type}, {corner.angle_deg:.0f} deg")

        for opening in photo.openings:
            short_id = opening.id.split("-")[-1]
            text = f"  opening {short_id}: {opening.type}"
            if opening.wall_letter:
                text += f" on wall {opening.wall_letter}"
            if opening.width_m:
                text += f", width {opening.width_m:.2f} m"
            lines.append(text)

        return "\n".join(lines)

    def build_messages(self, photos: list, overlay_paths: list) -> list:
        summaries = "\n".join(self.describe_photo(photo) for photo in photos)
        prompt = ROOM_LAYOUT_PROMPT.format(photo_count=len(photos), photo_summaries=summaries)

        content = [{"type": "text", "text": prompt}]

        for photo, overlay_path in zip(photos, overlay_paths):
            base64_image = self.image_converter.convert_to_base64_jpeg(overlay_path)
            content.append({"type": "text", "text": f"Photo {photo.photo_id}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        return [SystemMessage(content=ROOM_LAYOUT_SYSTEM_PROMPT), HumanMessage(content=content)]

    def find_layout(self, photos: list, overlay_paths: list) -> RoomLayout:
        messages = self.build_messages(photos, overlay_paths)

        self.vision_limiter.acquire()
        try:
            return self.llm_client.run(self.layout_model_name, messages, RoomLayout)
        finally:
            self.vision_limiter.release()
