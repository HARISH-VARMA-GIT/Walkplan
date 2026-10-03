import cv2
import numpy as np


WALL_COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
]
DOOR_COLOR = (255, 225, 25)
WINDOW_COLOR = (0, 255, 255)
CORNER_COLOR = (255, 255, 255)
CELL_SIZE = 4
TINT_STRENGTH = 0.35


class OverlayDrawer:

    def draw_text(self, image: np.ndarray, text: str, position: tuple, color: tuple, scale: float = 1.0):
        x, y = int(position[0]), int(position[1])
        thickness = max(int(2 * scale), 1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_walls(self, image: np.ndarray, analysis) -> np.ndarray:
        tint = image.copy()

        for index, wall_line in enumerate(analysis.wall_lines):
            color = WALL_COLORS[index % len(WALL_COLORS)]
            for row, col in zip(wall_line.plane.rows, wall_line.plane.cols):
                tint[row:row + CELL_SIZE, col:col + CELL_SIZE] = color

        return cv2.addWeighted(tint, TINT_STRENGTH, image, 1 - TINT_STRENGTH, 0)

    def draw_wall_letters(self, image: np.ndarray, analysis):
        scale = image.shape[1] / 500

        for index, wall_line in enumerate(analysis.wall_lines):
            color = WALL_COLORS[index % len(WALL_COLORS)]
            centre = (np.median(wall_line.plane.cols), np.median(wall_line.plane.rows))
            self.draw_text(image, wall_line.letter, centre, color, scale)

    def project(self, point_3d: np.ndarray, analysis):
        if point_3d[2] <= 0.05:
            return None

        intrinsics = analysis.intrinsics
        col = (intrinsics[0, 0] * point_3d[0] / point_3d[2] + intrinsics[0, 2]) * analysis.geometry.image_width
        row = (intrinsics[1, 1] * point_3d[1] / point_3d[2] + intrinsics[1, 2]) * analysis.geometry.image_height

        if 0 <= col < analysis.geometry.image_width and 0 <= row < analysis.geometry.image_height:
            return int(col), int(row)

        return None

    def draw_corners(self, image: np.ndarray, analysis):
        scale = image.shape[1] / 900
        height = 1.0 if analysis.geometry.floor_found else 0.0

        for corner in analysis.geometry.corners:
            point_3d = analysis.frame.to_3d(np.array(corner.point), height)
            pixel = self.project(point_3d, analysis)
            if pixel is None:
                continue

            cv2.circle(image, pixel, int(14 * scale) + 4, CORNER_COLOR, 3)
            self.draw_text(image, str(corner.number), (pixel[0] + 18, pixel[1]), CORNER_COLOR, scale)

    def draw_openings(self, image: np.ndarray, analysis):
        scale = image.shape[1] / 1100

        for opening in analysis.geometry.openings:
            color = DOOR_COLOR if opening.type == "door" else WINDOW_COLOR
            x1, y1, x2, y2 = [int(value) for value in opening.box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            short_id = opening.id.split("-")[-1]
            label = f"{short_id} {opening.type}"
            self.draw_text(image, label, (x1 + 6, y1 + int(30 * scale) + 6), color, scale)

    def draw_title(self, image: np.ndarray, analysis):
        scale = image.shape[1] / 1000
        self.draw_text(image, analysis.geometry.photo_id, (10, int(40 * scale) + 10), (255, 255, 255), scale * 1.2)

    def save(self, image_rgb: np.ndarray, analysis, output_path: str) -> str:
        image = self.draw_walls(image_rgb, analysis)
        self.draw_wall_letters(image, analysis)
        self.draw_corners(image, analysis)
        self.draw_openings(image, analysis)
        self.draw_title(image, analysis)

        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return output_path
