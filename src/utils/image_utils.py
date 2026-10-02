import base64
import io
import os

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class ImageConverter:

    def __init__(self, max_side_pixels: int = 2048, jpeg_quality: int = 90):
        self.max_side_pixels = max_side_pixels
        self.jpeg_quality = jpeg_quality
        self.image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".heic", ".heif"]

    def is_image_file(self, file_name: str) -> bool:
        extension = os.path.splitext(file_name)[1].lower()
        return extension in self.image_extensions

    def list_image_paths(self, folder_path: str) -> list[str]:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_paths = []

        for file_name in sorted(os.listdir(folder_path)):
            if self.is_image_file(file_name):
                image_paths.append(os.path.join(folder_path, file_name))

        return image_paths

    def convert_to_jpeg_bytes(self, image_path: str) -> bytes:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.thumbnail((self.max_side_pixels, self.max_side_pixels))

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality)
            return buffer.getvalue()

    def save_as_jpeg(self, image_path: str, output_path: str) -> str:
        jpeg_bytes = self.convert_to_jpeg_bytes(image_path)

        with open(output_path, "wb") as output_file:
            output_file.write(jpeg_bytes)

        return output_path

    def convert_to_base64_jpeg(self, image_path: str) -> str:
        jpeg_bytes = self.convert_to_jpeg_bytes(image_path)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
