import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

from models.floor_plan import CaptureInfo, FloorPlan
from models.image_models import RoomImageAnalysis, RoomLayout
from models.photo_geometry import RoomPhotoGeometry
from modules.floor_plan_generator.main import FloorPlanRenderer
from modules.image_processing.depth_estimator import DepthEstimator
from modules.image_processing.layout_reasoner import LayoutReasoner
from modules.image_processing.opening_finder import OpeningFinder, OpeningMeasurer
from modules.image_processing.overlay_drawer import OverlayDrawer
from modules.image_processing.photo_measurer import PhotoMeasurer
from modules.image_processing.room_assembler import RoomAssembler
from utils.image_utils import ImageConverter
from utils.llm_provider import LlmClient, VisionLimiter
from utils.prompts import ROOM_IMAGES_PROMPT, ROOM_IMAGES_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class OutputFolder:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.depth_folder = os.path.join(folder_path, "depth")
        self.overlay_folder = os.path.join(folder_path, "overlays")
        self.photo_geometry_path = os.path.join(folder_path, "photo_geometry.json")
        self.layout_path = os.path.join(folder_path, "layout.json")
        self.matches_path = os.path.join(folder_path, "wall_matches_used.json")
        self.floor_plan_path = os.path.join(folder_path, "floor_plan.json")
        self.svg_path = os.path.join(folder_path, "floor_plan.svg")

        os.makedirs(self.depth_folder, exist_ok=True)
        os.makedirs(self.overlay_folder, exist_ok=True)

    def depth_path(self, photo_id: str) -> str:
        return os.path.join(self.depth_folder, f"{photo_id}.npz")

    def overlay_path(self, photo_id: str) -> str:
        return os.path.join(self.overlay_folder, f"{photo_id}.jpg")

    def save_json(self, path: str, data: dict):
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2)

    def load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)


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
        self.vision_model_name = os.getenv("VISION_MODEL_NAME")

        self.depth_estimator = DepthEstimator()
        self.photo_measurer = PhotoMeasurer()
        self.opening_finder = OpeningFinder()
        self.opening_measurer = OpeningMeasurer()
        self.overlay_drawer = OverlayDrawer()
        self.room_assembler = RoomAssembler()
        self.renderer = FloorPlanRenderer()
        self.layout_reasoner = None

    def process_image(self, folder_path: str) -> dict:
        image_paths = self.image_converter.list_image_paths(folder_path)

        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")

        file_names = [os.path.basename(image_path) for image_path in image_paths]
        messages = self.build_messages(image_paths, file_names)

        self.vision_limiter.acquire()

        try:
            analysis = self.llm_client.run(self.vision_model_name, messages, RoomImageAnalysis)
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

    def photo_id(self, index: int) -> str:
        return f"P{index + 1:02d}"

    def estimate_all_depths(self, image_paths: list, output: OutputFolder):
        for index, image_path in enumerate(image_paths):
            photo_id = self.photo_id(index)
            logger.info("Depth for %s (%s)", photo_id, os.path.basename(image_path))

            image_rgb = self.image_converter.load_rgb_array(image_path)
            self.depth_estimator.estimate_with_cache(image_rgb, output.depth_path(photo_id))

        self.depth_estimator.unload_model()

    def measure_photos(self, room_name: str, image_paths: list, output: OutputFolder) -> RoomPhotoGeometry:
        self.estimate_all_depths(image_paths, output)
        photos = []

        for index, image_path in enumerate(image_paths):
            photo_id = self.photo_id(index)
            logger.info("Measuring %s (%s)", photo_id, os.path.basename(image_path))

            image_rgb = self.image_converter.load_rgb_array(image_path)
            depth = self.depth_estimator.estimate_with_cache(image_rgb, output.depth_path(photo_id))

            detections = self.opening_finder.find_openings(image_rgb)
            opening_pixels = self.opening_finder.combined_mask(detections, depth["mask"].shape)

            analysis = self.photo_measurer.measure(photo_id, image_path, depth, opening_pixels)
            analysis.geometry.openings = self.opening_measurer.measure(analysis, detections, depth)

            self.overlay_drawer.save(image_rgb, analysis, output.overlay_path(photo_id))
            photos.append(analysis.geometry)

        photo_set = RoomPhotoGeometry(room_name=room_name, photos=photos)
        output.save_json(output.photo_geometry_path, photo_set.model_dump())

        return photo_set

    def find_layout(self, photo_set: RoomPhotoGeometry, output: OutputFolder) -> RoomLayout:
        if self.layout_reasoner is None:
            self.layout_reasoner = LayoutReasoner()

        overlay_paths = [output.overlay_path(photo.photo_id) for photo in photo_set.photos]
        layout = self.layout_reasoner.find_layout(photo_set.photos, overlay_paths)
        output.save_json(output.layout_path, layout.model_dump())

        return layout

    def build_floor_plan(self, folder_path: str, output_folder: str, redo_photos: bool = False, redo_layout: bool = False) -> dict:
        image_paths = self.image_converter.list_image_paths(folder_path)
        if not image_paths:
            raise ValueError(f"No images found in {folder_path}")

        room_name = os.path.basename(os.path.normpath(folder_path))
        output = OutputFolder(output_folder)

        if redo_photos or not os.path.exists(output.photo_geometry_path):
            photo_set = self.measure_photos(room_name, image_paths, output)
            redo_layout = True
        else:
            logger.info("Using saved %s", output.photo_geometry_path)
            photo_set = RoomPhotoGeometry.model_validate(output.load_json(output.photo_geometry_path))

        if redo_layout or not os.path.exists(output.layout_path):
            logger.info("Asking the vision model for the room layout")
            layout = self.find_layout(photo_set, output)
        else:
            logger.info("Using saved %s", output.layout_path)
            layout = RoomLayout.model_validate(output.load_json(output.layout_path))

        assembled = self.room_assembler.assemble(photo_set, layout)
        for note in assembled["notes"]:
            logger.warning(note)

        logger.info("%d of %d photos placed in the room", assembled["photos_used"], len(photo_set.photos))
        output.save_json(output.matches_path, {"wall_matches": assembled["matches_used"]})

        plan = FloorPlan(
            capture=CaptureInfo(id=room_name, tier="photos"),
            rooms=[assembled["room"]],
            notes="; ".join(assembled["notes"]),
        )

        output.save_json(output.floor_plan_path, plan.model_dump(mode="json"))
        self.renderer.save_svg(plan, output.svg_path)

        return {
            "folder_path": folder_path,
            "output_folder": output_folder,
            "floor_plan": plan.model_dump(mode="json"),
            "svg_path": output.svg_path,
        }
