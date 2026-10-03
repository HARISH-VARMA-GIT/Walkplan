import cv2
import numpy as np

from measure_distance import DepthEstimator, DistanceMeasurer
from object_segmenter import ObjectFinder


IMAGE_PATH = "../inputs/images/room1/WhatsApp Image 2026-10-02 at 9.35.06 PM (1).jpeg"
OUTPUT_PATH = "../../output/room1/door_sam_measurement.png"
MODEL_NAME = "Ruicheng/moge-2-vitl-normal"
OBJECT_NAME = "door"
WINDOW_SIZE = 5
BAND_FRACTION = 0.2
SAMPLE_STEP = 5


class MaskMeasurer:
    def __init__(self, distance_measurer):
        self.distance_measurer = distance_measurer

    def measure_height(self, mask):
        ys, xs = np.where(mask)
        center_x = (xs.min() + xs.max()) // 2
        half_band = max(int((xs.max() - xs.min()) * BAND_FRACTION / 2), 1)

        lengths = []
        best_line = None

        for x in range(center_x - half_band, center_x + half_band + 1, SAMPLE_STEP):
            column_ys = np.where(mask[:, x])[0]
            if len(column_ys) == 0:
                continue

            top = (x, int(column_ys.min()))
            bottom = (x, int(column_ys.max()))

            try:
                length = self.distance_measurer.measure(top, bottom)
            except ValueError:
                continue

            lengths.append(length)
            if x == center_x or best_line is None:
                best_line = (top, bottom)

        return self.summarize(lengths), best_line

    def measure_width(self, mask):
        ys, xs = np.where(mask)
        center_y = (ys.min() + ys.max()) // 2
        half_band = max(int((ys.max() - ys.min()) * BAND_FRACTION / 2), 1)

        lengths = []
        best_line = None

        for y in range(center_y - half_band, center_y + half_band + 1, SAMPLE_STEP):
            row_xs = np.where(mask[y, :])[0]
            if len(row_xs) == 0:
                continue

            left = (int(row_xs.min()), y)
            right = (int(row_xs.max()), y)

            try:
                length = self.distance_measurer.measure(left, right)
            except ValueError:
                continue

            lengths.append(length)
            if y == center_y or best_line is None:
                best_line = (left, right)

        return self.summarize(lengths), best_line

    def summarize(self, lengths):
        if len(lengths) == 0:
            return None

        return float(np.median(lengths))


class ResultDrawer:
    def __init__(self, image_rgb):
        self.image_rgb = image_rgb

    def save(self, mask, box, height, height_line, width, width_line, output_path):
        image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)

        overlay = image_bgr.copy()
        overlay[mask] = (0, 200, 0)
        image_bgr = cv2.addWeighted(overlay, 0.4, image_bgr, 0.6, 0)

        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        self.draw_line(image_bgr, height_line, height, (0, 0, 255))
        self.draw_line(image_bgr, width_line, width, (0, 165, 255))

        cv2.imwrite(output_path, image_bgr)

    def draw_line(self, image_bgr, line, length, color):
        if line is None or length is None:
            return

        start, end = line
        cv2.line(image_bgr, start, end, color, 3)

        middle = ((start[0] + end[0]) // 2 + 10, (start[1] + end[1]) // 2)
        cv2.putText(image_bgr, f"{length:.2f} m", middle, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


class AutoMeasureApp:
    def __init__(self, image_path, output_path, object_name):
        self.image_path = image_path
        self.output_path = output_path
        self.object_name = object_name

    def run(self):
        image_rgb = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)

        print("Estimating depth...")
        estimator = DepthEstimator(MODEL_NAME)
        points, depth_mask = estimator.estimate(image_rgb)

        print(f"Looking for '{self.object_name}'...")
        finder = ObjectFinder()
        found = finder.find_object_mask(image_rgb, self.object_name)

        if found is None:
            print(f"No '{self.object_name}' found. Try another word or a lower threshold.")
            return

        print(f"Found with score {found['score']:.2f}")

        distance_measurer = DistanceMeasurer(points, depth_mask, WINDOW_SIZE)
        mask_measurer = MaskMeasurer(distance_measurer)

        height, height_line = mask_measurer.measure_height(found["mask"])
        width, width_line = mask_measurer.measure_width(found["mask"])

        print(f"Height: {height} m")
        print(f"Width: {width} m")

        drawer = ResultDrawer(image_rgb)
        drawer.save(found["mask"], found["box"], height, height_line, width, width_line, self.output_path)
        print(f"Saved {self.output_path}")


if __name__ == "__main__":
    app = AutoMeasureApp(IMAGE_PATH, OUTPUT_PATH, OBJECT_NAME)
    app.run()
