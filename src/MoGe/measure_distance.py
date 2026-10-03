import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from moge.model.v2 import MoGeModel


IMAGE_PATH = "../inputs/images/room1/WhatsApp Image 2026-10-02 at 9.35.06 PM.jpeg"
OUTPUT_PATH = "../../output/room1/door_measurement.png"
MODEL_NAME = "Ruicheng/moge-2-vitl-normal"
WINDOW_SIZE = 5


class DepthEstimator:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(model_name).to(self.device).eval()

    def estimate(self, image_rgb):
        image_tensor = torch.tensor(image_rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        output = self.model.infer(image_tensor)

        points = output["points"].cpu().numpy()
        mask = output["mask"].cpu().numpy()

        return points, mask


class PointPicker:
    def __init__(self, image_rgb):
        self.image_rgb = image_rgb

    def pick_points(self, count):
        figure = plt.figure(figsize=(12, 8))
        plt.imshow(self.image_rgb)
        plt.title(f"Click {count} points, then wait")

        clicks = plt.ginput(count, timeout=0)
        plt.close(figure)

        pixels = []
        for x, y in clicks:
            pixels.append((int(round(x)), int(round(y))))

        return pixels


class DistanceMeasurer:
    def __init__(self, points, mask, window_size):
        self.points = points
        self.mask = mask
        self.half = window_size // 2

    def get_point_3d(self, pixel):
        u, v = pixel
        height, width = self.mask.shape

        top = max(v - self.half, 0)
        bottom = min(v + self.half + 1, height)
        left = max(u - self.half, 0)
        right = min(u + self.half + 1, width)

        window_points = self.points[top:bottom, left:right].reshape(-1, 3)
        window_mask = self.mask[top:bottom, left:right].reshape(-1)

        valid = window_mask & np.isfinite(window_points).all(axis=1)
        if not valid.any():
            raise ValueError(f"No valid depth near pixel {pixel}. Click somewhere else.")

        return np.median(window_points[valid], axis=0)

    def measure(self, pixel_a, pixel_b):
        point_a = self.get_point_3d(pixel_a)
        point_b = self.get_point_3d(pixel_b)

        return float(np.linalg.norm(point_a - point_b))


class MeasurementDrawer:
    def __init__(self, image_rgb):
        self.image_rgb = image_rgb

    def save(self, pixel_a, pixel_b, length_meters, output_path):
        image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)

        cv2.line(image_bgr, pixel_a, pixel_b, (0, 0, 255), 3)
        cv2.circle(image_bgr, pixel_a, 8, (0, 255, 0), -1)
        cv2.circle(image_bgr, pixel_b, 8, (0, 255, 0), -1)

        label = f"{length_meters:.2f} m"
        middle = ((pixel_a[0] + pixel_b[0]) // 2 + 10, (pixel_a[1] + pixel_b[1]) // 2)
        cv2.putText(image_bgr, label, middle, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imwrite(output_path, image_bgr)


class MeasureApp:
    def __init__(self, image_path, output_path, model_name):
        self.image_path = image_path
        self.output_path = output_path
        self.model_name = model_name

    def run(self):
        image_rgb = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)

        print("Estimating depth...")
        estimator = DepthEstimator(self.model_name)
        points, mask = estimator.estimate(image_rgb)

        measurer = DistanceMeasurer(points, mask, WINDOW_SIZE)
        picker = PointPicker(image_rgb)
        drawer = MeasurementDrawer(image_rgb)

        while True:
            pixels = picker.pick_points(2)
            if len(pixels) < 2:
                break

            length_meters = measurer.measure(pixels[0], pixels[1])
            print(f"{pixels[0]} -> {pixels[1]}: {length_meters:.3f} m ({length_meters * 100:.1f} cm)")

            drawer.save(pixels[0], pixels[1], length_meters, self.output_path)
            print(f"Saved {self.output_path}")

            answer = input("Measure another? (y/n): ")
            if answer.strip().lower() != "y":
                break


if __name__ == "__main__":
    app = MeasureApp(IMAGE_PATH, OUTPUT_PATH, MODEL_NAME)
    app.run()
