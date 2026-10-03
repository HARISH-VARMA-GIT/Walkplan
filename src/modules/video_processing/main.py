import json
import logging
import os

from models.floor_plan import CaptureInfo, FloorPlan
from models.video_models import VideoFrame
from modules.image_processing.main import ImageProcessingService
from modules.video_processing.audio_extractor import AudioExtractor
from modules.video_processing.damage_finder import DamageFinder
from modules.video_processing.frame_selector import FrameSampler, FrameSelector
from modules.video_processing.room_aligner import ChunkAligner
from modules.video_processing.room_combiner import RoomCombiner
from modules.video_processing.room_splitter import RoomSplitter
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
        self.rooms_path = os.path.join(folder_path, "rooms.json")
        self.candidate_folder = os.path.join(folder_path, "frame_candidates")
        self.candidates_path = os.path.join(folder_path, "frame_candidates.json")
        self.frame_folder = os.path.join(folder_path, "frames")
        self.frames_path = os.path.join(folder_path, "frames.json")
        self.floor_plan_path = os.path.join(folder_path, "floor_plan.json")
        self.svg_path = os.path.join(folder_path, "floor_plan.svg")

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
        self.room_splitter = None
        self.frame_sampler = FrameSampler(self.ffmpeg)
        self.frame_selector = FrameSelector()
        self.image_service = ImageProcessingService()
        self.aligner = ChunkAligner()
        self.room_combiner = RoomCombiner()

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

    def find_rooms(self, transcript: dict, duration: float, room_name: str, output: VideoOutputFolder, redo: bool) -> list:
        if not redo and os.path.exists(output.rooms_path):
            logger.info("Using saved %s", output.rooms_path)
            return output.load_json(output.rooms_path)["rooms"]

        if self.room_splitter is None:
            self.room_splitter = RoomSplitter()

        logger.info("Finding the rooms in the transcript")
        result = self.room_splitter.split(transcript["segments"], duration, room_name)
        output.save_json(output.rooms_path, result)

        return result["rooms"]

    def sample_candidates(self, video_file: str, output: VideoOutputFolder, redo: bool) -> list:
        if not redo and os.path.exists(output.candidates_path):
            return output.load_json(output.candidates_path)

        logger.info("Sampling candidate frames from the video")
        candidates = self.frame_sampler.sample(video_file, output.candidate_folder)
        output.save_json(output.candidates_path, candidates)

        return candidates

    def select_frames(self, video_file: str, damages: list, duration: float, max_frames: int, output: VideoOutputFolder,
                      frame_output: VideoOutputFolder, redo: bool, start: float = 0.0, end: float = None, transitions: list = None,
                      candidates: list = None) -> tuple:
        if not redo and os.path.exists(frame_output.frames_path):
            logger.info("Using saved %s", frame_output.frames_path)
            frames = [VideoFrame.model_validate(item) for item in frame_output.load_json(frame_output.frames_path)]
            return frames, False

        if candidates is None:
            candidates = self.sample_candidates(video_file, output, redo)
        selected = self.frame_selector.select(candidates, damages, duration, max_frames, start, end, transitions)
        frames = self.frame_selector.save(selected, frame_output.frame_folder)
        frame_output.save_json(frame_output.frames_path, [frame.model_dump() for frame in frames])

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

    def damage_details(self, frames: list, damages: list, folder_prefix: str = "") -> dict:
        details = {}
        for damage in damages:
            source_frames = []
            for frame in frames:
                if damage["id"] in frame.damage_ids:
                    source_frames.append(f"{folder_prefix}damage/{damage['id']}_{frame.photo_id}.jpg")

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

        rooms = self.find_rooms(transcript, duration, room_name, output, redo_video)
        logger.info("Rooms: %s", [f"{room['name']} ({room['start_seconds']}-{room['end_seconds']}s)" for room in rooms])

        if len(rooms) == 1:
            if room_type == "other":
                room_type = rooms[0]["room_type"]
            result = self.build_one_room(video_file, room_name, output, damages, duration, room_type, max_frames,
                                         redo_video, redo_photos, redo_layout)
        else:
            result = self.build_many_rooms(video_file, room_name, output, damages, rooms, duration, max_frames,
                                           redo_video, redo_photos, redo_layout)

        result["video_path"] = video_file
        result["damages"] = []
        for room in result["floor_plan"]["rooms"]:
            result["damages"].extend(room["damage"])

        return result

    def build_one_room(self, video_file: str, room_name: str, output: VideoOutputFolder, damages: list, duration: float,
                       room_type: str, max_frames: int, redo_video: bool, redo_photos: bool, redo_layout: bool) -> dict:
        frames, frames_changed = self.select_frames(video_file, damages, duration, max_frames, output, output, redo_video)
        image_paths = [os.path.join(output.frame_folder, frame.file_name) for frame in frames]

        return self.image_service.build_floor_plan_from_paths(
            image_paths,
            room_name,
            output.folder_path,
            redo_photos=redo_photos or frames_changed,
            redo_layout=redo_layout,
            layout_mode="geometry",
            room_type=room_type,
            capture_tier="video",
            damage_requests=self.damage_requests(frames, damages),
            damage_details=self.damage_details(frames, damages),
            shared_fov=True,
        )

    def folder_name(self, room: dict) -> str:
        safe_name = ""
        for character in room["name"].lower():
            safe_name += character if character.isalnum() else "_"
        return f"{room['index']:02d}_{safe_name.strip('_')}"

    def shared_pairs(self, result_before: dict, result_after: dict) -> list:
        pose_by_time = {}
        for frame, pose in zip(result_before["frames"], result_before["poses"]):
            pose_by_time[frame.time_seconds] = pose

        pairs = []
        for frame, pose in zip(result_after["frames"], result_after["poses"]):
            if frame.time_seconds in pose_by_time:
                pairs.append((pose_by_time[frame.time_seconds], pose))

        return pairs

    def build_room_in_video(self, video_file: str, output: VideoOutputFolder, room: dict, damages: list, transitions: list, candidates: list,
                            duration: float, max_frames: int, redo_video: bool, redo_photos: bool, redo_layout: bool) -> dict:
        folder_name = self.folder_name(room)
        room_output = VideoOutputFolder(os.path.join(output.folder_path, "rooms", folder_name))

        room_damages = []
        for damage in damages:
            if room["start_seconds"] <= damage["start_seconds"] < room["end_seconds"]:
                room_damages.append(damage)

        frames, frames_changed = self.select_frames(video_file, room_damages, duration, max_frames, output, room_output, redo_video,
                                                    room["start_seconds"], room["end_seconds"], transitions, candidates)
        image_paths = [os.path.join(room_output.frame_folder, frame.file_name) for frame in frames]

        result = self.image_service.build_floor_plan_from_paths(
            image_paths,
            room["name"],
            room_output.folder_path,
            redo_photos=redo_photos or frames_changed,
            redo_layout=redo_layout or frames_changed,
            layout_mode="geometry",
            room_type=room["room_type"],
            capture_tier="video",
            damage_requests=self.damage_requests(frames, room_damages),
            damage_details=self.damage_details(frames, room_damages, f"rooms/{folder_name}/"),
            shared_fov=True,
            pose_only_photo_ids=[frame.photo_id for frame in frames if frame.purpose == "transition"],
        )

        result.update({"index": room["index"], "name": room["name"], "room_type": room["room_type"], "frames": frames})
        return result

    def build_many_rooms(self, video_file: str, video_name: str, output: VideoOutputFolder, damages: list, rooms: list,
                         duration: float, max_frames: int, redo_video: bool, redo_photos: bool, redo_layout: bool) -> dict:
        candidates = self.sample_candidates(video_file, output, redo_video)

        transitions = []
        for room in rooms[1:]:
            transitions.append(self.frame_selector.transition_frames(candidates, room["start_seconds"]))

        results = []
        for position, room in enumerate(rooms):
            room_transitions = []
            if position > 0:
                room_transitions += transitions[position - 1]
            if position < len(rooms) - 1:
                room_transitions += transitions[position]

            logger.info("Room %d of %d: %s", position + 1, len(rooms), room["name"])
            results.append(self.build_room_in_video(video_file, output, room, damages, room_transitions, candidates, duration, max_frames,
                                                    redo_video, redo_photos, redo_layout))

        transforms = [self.aligner.identity()]
        notes = []
        for position in range(1, len(results)):
            aligned = self.aligner.align(self.shared_pairs(results[position - 1], results[position]))
            notes.append(f"{results[position - 1]['name']} -> {results[position]['name']}: {aligned['note']}")
            logger.info(notes[-1])

            if aligned["transform"] is None or transforms[-1] is None:
                transforms.append(None)
            else:
                transforms.append(aligned["transform"].then(transforms[-1]))

        combined = self.room_combiner.combine(results, transforms)
        for note in combined["notes"]:
            logger.info(note)
        notes += combined["notes"]

        plan = FloorPlan(
            capture=CaptureInfo(id=video_name, tier="video"),
            rooms=combined["rooms"],
            adjacency=combined["adjacency"],
            notes="; ".join(notes),
        )

        output.save_json(output.floor_plan_path, plan.model_dump(mode="json"))
        self.image_service.renderer.save_svg(plan, output.svg_path)

        return {
            "output_folder": output.folder_path,
            "floor_plan": plan.model_dump(mode="json"),
            "svg_path": output.svg_path,
        }
