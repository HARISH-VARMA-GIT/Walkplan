import argparse
import logging
import os

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class App:

    def read_arguments(self):
        parser = argparse.ArgumentParser(description="Build a floor plan from room photos")
        parser.add_argument("--images", required=True, help="Folder with the photos of one room")
        parser.add_argument("--output", default=None, help="Output folder (default: output/<room name>)")
        parser.add_argument("--redo-photos", action="store_true", help="Measure the photos again (depth is still cached)")
        parser.add_argument("--redo-layout", action="store_true", help="Ask the vision model for the layout again")
        return parser.parse_args()

    def run(self):
        from modules.image_processing.main import ImageProcessingService

        arguments = self.read_arguments()
        room_name = os.path.basename(os.path.normpath(arguments.images))
        output_folder = arguments.output or os.path.join(PROJECT_FOLDER, "output", room_name)

        service = ImageProcessingService()
        result = service.build_floor_plan(arguments.images, output_folder, arguments.redo_photos, arguments.redo_layout)

        logger.info("Floor plan saved to %s", result["svg_path"])


if __name__ == "__main__":
    App().run()
