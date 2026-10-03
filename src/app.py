import argparse
import logging
import os

from dotenv import load_dotenv


PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_FOLDER, ".cache", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_FOLDER, ".cache", "torch"))

from modules.image_processing.main import ImageProcessingService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)


class App:

    def read_arguments(self):
        parser = argparse.ArgumentParser(description="Build a floor plan from room photos")
        parser.add_argument("--images", required=True, help="Folder with the photos of one room")
        parser.add_argument("--output", default=None, help="Output folder (default: output/<room name>)")
        parser.add_argument("--redo-photos", action="store_true", help="Measure the photos again (depth is still cached)")
        parser.add_argument("--redo-layout", action="store_true", help="Ask the vision model for the layout again")
        parser.add_argument("--layout", choices=["geometry", "llm"], default="geometry",
                            help="geometry: walls from MapAnything camera poses (default). llm: vision model guesses the layout")
        parser.add_argument("--room-type", default="other", help="Room label for the plan, e.g. bedroom")
        return parser.parse_args()

    def run(self):

        arguments = self.read_arguments()
        room_name = os.path.basename(os.path.normpath(arguments.images))
        output_folder = arguments.output or os.path.join(PROJECT_FOLDER, "output", room_name)

        service = ImageProcessingService()
        result = service.build_floor_plan(arguments.images, output_folder, arguments.redo_photos, arguments.redo_layout,
                                          arguments.layout, arguments.room_type)

        logger.info("Floor plan saved to %s", result["svg_path"])


if __name__ == "__main__":
    App().run()
