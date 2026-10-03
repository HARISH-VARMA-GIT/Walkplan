import os
import re
import shutil
import subprocess


class FfmpegRunner:

    def __init__(self):
        self.ffmpeg_path = self.find_ffmpeg()

    def find_ffmpeg(self) -> str:
        configured = os.getenv("FFMPEG_PATH")
        if configured:
            return configured

        on_path = shutil.which("ffmpeg")
        if on_path:
            return on_path

        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError("ffmpeg not found. Install it (winget install ffmpeg) or pip install imageio-ffmpeg, or set FFMPEG_PATH")

    def run(self, arguments: list) -> str:
        command = [self.ffmpeg_path, "-hide_banner", "-y"] + arguments
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")

        if result.returncode != 0:
            last_lines = "\n".join(result.stderr.strip().splitlines()[-10:])
            raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{last_lines}")

        return result.stderr

    def describe(self, video_path: str) -> str:
        command = [self.ffmpeg_path, "-hide_banner", "-i", video_path]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return result.stderr

    def duration_seconds(self, video_path: str) -> float:
        found = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", self.describe(video_path))
        if found is None:
            return 0.0

        hours, minutes, seconds = found.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def has_audio(self, video_path: str) -> bool:
        return re.search(r"Stream #.*Audio:", self.describe(video_path)) is not None
