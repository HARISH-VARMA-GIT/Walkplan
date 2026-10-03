import argparse
import logging
import os

from dotenv import load_dotenv


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_FOLDER, ".cache", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_FOLDER, ".cache", "torch"))

from modules.image_processing.main import ImageProcessingService
from modules.lidar_processing.main import LidarProcessingService
from modules.video_processing.main import VideoProcessingService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)


class App:

    def read_arguments(self):
        parser = argparse.ArgumentParser(description="Build a floor plan from room photos, a room video or a LiDAR scan")
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--images", help="Folder with the photos of one room")
        source.add_argument("--video", help="Video file of one room, or a folder holding it")
        source.add_argument("--lidar", help="Folder with one Stray Scanner LiDAR capture (one or several rooms)")
        parser.add_argument("--output", default=None,
                            help="Output folder (default: output/<name>, output/<name>_video for videos, output/<name>_lidar for LiDAR)")
        parser.add_argument("--redo-photos", action="store_true", help="Measure the photos again (depth is still cached)")
        parser.add_argument("--redo-layout", action="store_true", help="Build the room layout again")
        parser.add_argument("--redo-video", action="store_true", help="Video only: transcribe, find damage and pick frames again")
        parser.add_argument("--redo-lidar", action="store_true", help="LiDAR only: fuse the depth frames and look for doors and windows again")
        parser.add_argument("--no-openings", action="store_true", help="LiDAR only: skip the camera search for doors and windows (faster, no GPU)")
        parser.add_argument("--max-frames", type=int, default=24, help="Video only: how many frames to use (default 24)")
        parser.add_argument("--layout", choices=["geometry", "llm"], default="geometry",
                            help="Photos only. geometry: walls from MapAnything camera poses (default). llm: vision model guesses the layout")
        parser.add_argument("--room-type", default="other", help="Room label for the plan, e.g. bedroom")
        return parser.parse_args()

    def run_images(self, arguments):
        room_name = os.path.basename(os.path.normpath(arguments.images))
        output_folder = arguments.output or os.path.join(PROJECT_FOLDER, "output", room_name)

        service = ImageProcessingService()
        result = service.build_floor_plan(arguments.images, output_folder, arguments.redo_photos, arguments.redo_layout,
                                          arguments.layout, arguments.room_type)

        logger.info("Floor plan saved to %s", result["svg_path"])

    def run_video(self, arguments):
        service = VideoProcessingService()
        room_name = service.room_name_for(arguments.video)
        output_folder = arguments.output or os.path.join(PROJECT_FOLDER, "output", f"{room_name}_video")

        result = service.build_floor_plan(arguments.video, output_folder, arguments.room_type, arguments.max_frames,
                                          arguments.redo_video, arguments.redo_photos, arguments.redo_layout)

        for damage in result["damages"]:
            logger.info("Damage %s: %s on %s %s, offset %s m, height %s m (%s)", damage["id"], damage["damage_class"],
                        damage["surface_type"], damage["wall_id"] or "", damage["offset_m"], damage["height_m"], damage["location_method"])

        logger.info("Floor plan saved to %s", result["svg_path"])

    def run_lidar(self, arguments):
        service = LidarProcessingService()
        name = service.room_name_for(arguments.lidar)
        output_folder = arguments.output or os.path.join(PROJECT_FOLDER, "output", f"{name}_lidar")

        result = service.build_floor_plan(arguments.lidar, output_folder, arguments.room_type, arguments.redo_lidar,
                                          not arguments.no_openings)

        for room in result["floor_plan"]["rooms"]:
            ceiling = room["ceiling_height"]["value"] if room["ceiling_height"] else None
            logger.info("%s: %d walls, %.2f m2, ceiling %s m, %d openings", room["id"], len(room["walls"]),
                        room["floor_area"]["value"], ceiling, len(room["openings"]))

        logger.info("Floor plan saved to %s", result["svg_path"])

    def run(self):
        arguments = self.read_arguments()

        if arguments.lidar:
            self.run_lidar(arguments)
        elif arguments.video:
            self.run_video(arguments)
        else:
            self.run_images(arguments)


if __name__ == "__main__":
    App().run()
