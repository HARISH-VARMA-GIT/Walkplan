import logging

from utils.ffmpeg_runner import FfmpegRunner


logger = logging.getLogger(__name__)


class AudioExtractor:

    def __init__(self, ffmpeg: FfmpegRunner):
        self.ffmpeg = ffmpeg

    def extract(self, video_path: str, output_path: str):
        if not self.ffmpeg.has_audio(video_path):
            logger.warning("Video has no audio track, no damage can be found from speech")
            return None

        self.ffmpeg.run(["-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", output_path])
        return output_path
