import json
import logging
import os

from models.video_models import VideoFrame
from modules.image_processing.main import ImageProcessingService
from modules.video_processing.audio_extractor import AudioExtractor
from modules.video_processing.damage_finder import DamageFinder
from modules.video_processing.frame_selector import FrameSampler, FrameSelector
from modules.video_processing.transcriber import Transcriber
from utils.ffmpeg_runner import FfmpegRunner


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi", ".mkv"]
DEFAULT_MAX_FRAMES = 24


class VideoOutputFolder:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.audio_path = os.path.join(folder_path, "audio.m4a")
        self.transcript_path = os.path.join(folder_path, "transcript.json")
        self.damage_mentions_path = os.path.join(folder_path, "damage_mentions.json")
        self.candidate_folder = os.path.join(folder_path, "frame_candidates")
        self.candidates_path = os.path.join(folder_path, "frame_candidates.json")
        self.frame_folder = os.path.join(folder_path, "frames")
        self.frames_path = os.path.join(folder_path, "frames.json")

        os.makedirs(folder_path, exist_ok=True)

    def save_json(self, path: str, data):
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)

    def load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)


class VideoProcessingService:

    def __init__(self):
        self.ffmpeg = FfmpegRunner()
        self.audio_extractor = AudioExtractor(self.ffmpeg)
        self.transcriber = Transcriber()
        self.damage_finder = None
        self.frame_sampler = FrameSampler(self.ffmpeg)
        self.frame_selector = FrameSelector()
        self.image_service = ImageProcessingService()

    def find_video(self, video_path: str) -> str:
        if os.path.isfile(video_path):
            return video_path

        if not os.path.isdir(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        for file_name in sorted(os.listdir(video_path)):
            if os.path.splitext(file_name)[1].lower() in VIDEO_EXTENSIONS:
                return os.path.join(video_path, file_name)

        raise FileNotFoundError(f"No video file in {video_path}")

    def room_name_for(self, video_path: str) -> str:
        if os.path.isdir(video_path):
            return os.path.basename(os.path.normpath(video_path))
        return os.path.splitext(os.path.basename(video_path))[0]

    def transcribe(self, video_file: str, output: VideoOutputFolder, redo: bool) -> dict:
        if not redo and os.path.exists(output.transcript_path):
            logger.info("Using saved %s", output.transcript_path)
            return output.load_json(output.transcript_path)

        logger.info("Extracting audio")
        audio_path = self.audio_extractor.extract(video_file, output.audio_path)

        if audio_path is None:
            transcript = {"model": None, "language": None, "text": "", "segments": [], "words": []}
        else:
            logger.info("Transcribing audio (%s)", self.transcriber.model_name)
            transcript = self.transcriber.transcribe(audio_path)

        output.save_json(output.transcript_path, transcript)
        return transcript

    def find_damages(self, transcript: dict, duration: float, output: VideoOutputFolder, redo: bool) -> list:
        if not redo and os.path.exists(output.damage_mentions_path):
            logger.info("Using saved %s", output.damage_mentions_path)
            return output.load_json(output.damage_mentions_path)["damages"]

        if self.damage_finder is None:
            self.damage_finder = DamageFinder()

        logger.info("Finding damage in the transcript")
        result = self.damage_finder.find(transcript["segments"], duration)
        output.save_json(output.damage_mentions_path, result)

        return result["damages"]

    def sample_candidates(self, video_file: str, output: VideoOutputFolder, redo: bool) -> list:
        if not redo and os.path.exists(output.candidates_path):
            return output.load_json(output.candidates_path)

        logger.info("Sampling candidate frames from the video")
        candidates = self.frame_sampler.sample(video_file, output.candidate_folder)
        output.save_json(output.candidates_path, candidates)

        return candidates

    def select_frames(self, video_file: str, damages: list, duration: float, max_frames: int,
                      output: VideoOutputFolder, redo: bool) -> tuple:
        if not redo and os.path.exists(output.frames_path):
            logger.info("Using saved %s", output.frames_path)
            frames = [VideoFrame.model_validate(item) for item in output.load_json(output.frames_path)]
            return frames, False

        candidates = self.sample_candidates(video_file, output, redo)
        selected = self.frame_selector.select(candidates, damages, duration, max_frames)
        frames = self.frame_selector.save(selected, output.frame_folder)
        output.save_json(output.frames_path, [frame.model_dump() for frame in frames])

        damage_frame_count = len([frame for frame in frames if frame.damage_ids])
        logger.info("Picked %d frames (%d show damage)", len(frames), damage_frame_count)

        return frames, True

    def damage_requests(self, frames: list, damages: list) -> dict:
        damage_by_id = {}
        for damage in damages:
            damage_by_id[damage["id"]] = damage

        requests = {}
        for frame in frames:
            for damage_id in frame.damage_ids:
                if damage_id not in damage_by_id:
                    continue

                damage = damage_by_id[damage_id]
                requests.setdefault(frame.photo_id, []).append({
                    "damage_id": damage_id,
                    "search_terms": damage["search_terms"],
                    "surface_hint": damage["surface_type"],
                    "damage_class": damage["damage_class"],
                    "description": damage["description"],
                    "quote": damage["quote"],
                })

        return requests

    def damage_details(self, frames: list, damages: list) -> dict:
        details = {}
        for damage in damages:
            source_frames = []
            for frame in frames:
                if damage["id"] in frame.damage_ids:
                    source_frames.append(f"damage/{damage['id']}_{frame.photo_id}.jpg")

            details[damage["id"]] = dict(damage)
            details[damage["id"]]["source_frames"] = source_frames

        return details

    def build_floor_plan(self, video_path: str, output_folder: str, room_type: str = "other", max_frames: int = DEFAULT_MAX_FRAMES,
                         redo_video: bool = False, redo_photos: bool = False, redo_layout: bool = False) -> dict:
        video_file = self.find_video(video_path)
        room_name = self.room_name_for(video_path)
        output = VideoOutputFolder(output_folder)

        duration = self.ffmpeg.duration_seconds(video_file)
        logger.info("Video %s: %.1f s", os.path.basename(video_file), duration)

        transcript = self.transcribe(video_file, output, redo_video)
        damages = self.find_damages(transcript, duration, output, redo_video)
        logger.info("Damage mentioned: %s", [f"{damage['id']} {damage['damage_class']} at {damage['start_seconds']}s" for damage in damages])

        frames, frames_changed = self.select_frames(video_file, damages, duration, max_frames, output, redo_video)
        if frames_changed:
            redo_photos = True

        image_paths = [os.path.join(output.frame_folder, frame.file_name) for frame in frames]

        result = self.image_service.build_floor_plan_from_paths(
            image_paths,
            room_name,
            output_folder,
            redo_photos=redo_photos,
            redo_layout=redo_layout,
            layout_mode="geometry",
            room_type=room_type,
            capture_tier="video",
            damage_requests=self.damage_requests(frames, damages),
            damage_details=self.damage_details(frames, damages),
            shared_fov=True,
        )

        result["video_path"] = video_file
        result["damages"] = result["floor_plan"]["rooms"][0]["damage"]
        return result
