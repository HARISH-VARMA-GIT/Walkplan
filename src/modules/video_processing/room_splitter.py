import os

from langchain_core.messages import HumanMessage, SystemMessage

from models.video_models import RoomMentionList
from utils.llm_provider import LlmClient
from utils.prompts import ROOM_SPLIT_PROMPT, ROOM_SPLIT_SYSTEM_PROMPT


MIN_ROOM_SECONDS = 8.0


class RoomSplitter:

    def __init__(self):
        self.llm_client = LlmClient(
            model_env_name="LLM_DAMAGE_MODEL_CHAIN",
            default_models="gpt-4.1,gpt-4o",
            timeout=120,
        )
        self.model_name = os.getenv("DAMAGE_MODEL_NAME")

    def format_transcript(self, segments: list) -> str:
        lines = []
        for segment in segments:
            lines.append(f"[{segment['start_seconds']:.1f}s - {segment['end_seconds']:.1f}s] {segment['text']}")
        return "\n".join(lines)

    def one_room(self, room_name: str, duration: float) -> list:
        return [{"index": 1, "name": room_name, "room_type": "other", "start_seconds": 0.0, "end_seconds": duration}]

    def split(self, segments: list, duration: float, room_name: str) -> dict:
        if not segments:
            return {"rooms": self.one_room(room_name, duration)}

        prompt = ROOM_SPLIT_PROMPT.format(duration=duration, transcript=self.format_transcript(segments))
        messages = [SystemMessage(content=ROOM_SPLIT_SYSTEM_PROMPT), HumanMessage(content=prompt)]

        result = self.llm_client.run(self.model_name, messages, RoomMentionList)
        rooms = self.clean_up([mention.model_dump() for mention in result.rooms], duration)

        if not rooms:
            rooms = self.one_room(room_name, duration)

        return {"rooms": rooms}

    def start_of(self, room: dict) -> float:
        return room["start_seconds"]

    def clean_up(self, rooms: list, duration: float) -> list:
        kept = []

        for room in sorted(rooms, key=self.start_of):
            room["start_seconds"] = round(min(max(room["start_seconds"], 0.0), duration), 2)

            if kept and kept[-1]["name"].lower() == room["name"].lower():
                continue
            kept.append(room)

        if kept:
            kept[0]["start_seconds"] = 0.0

        for index, room in enumerate(kept):
            room["end_seconds"] = kept[index + 1]["start_seconds"] if index + 1 < len(kept) else round(duration, 2)

        long_enough = []
        for room in kept:
            if room["end_seconds"] - room["start_seconds"] >= MIN_ROOM_SECONDS or not long_enough:
                long_enough.append(room)
            else:
                long_enough[-1]["end_seconds"] = room["end_seconds"]

        for index, room in enumerate(long_enough):
            room["index"] = index + 1

        return long_enough
