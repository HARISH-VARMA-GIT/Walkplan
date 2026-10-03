import json
import os
import sys

SRC_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_FOLDER)

from dotenv import load_dotenv

load_dotenv()

from modules.image_processing.main import ImageProcessingService


PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
IMAGES_FOLDER = os.path.join(PROJECT_FOLDER, "data", "images")
OUTPUT_FILE = os.path.join(PROJECT_FOLDER, "data", "image_results.json")


def run_test():
    service = ImageProcessingService()
    result = service.process_image(IMAGES_FOLDER)

    with open(OUTPUT_FILE, "w") as output_file:
        json.dump(result, output_file, indent=4)

    print(json.dumps(result, indent=4))
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_test()
