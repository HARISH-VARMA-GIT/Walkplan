import os

from langchain_core.messages import HumanMessage, SystemMessage

from models.video_models import DamageMentionList
from utils.llm_provider import LlmClient
from utils.prompts import DAMAGE_PROMPT, DAMAGE_SYSTEM_PROMPT


MIN_DURATION_S = 1.0
SAME_DAMAGE_GAP_S = 2.0
MAX_SEARCH_TERMS = 3

DEFAULT_SEARCH_TERMS = {
    "water_stain": ["water stain", "stain"],
    "mold": ["mold", "black spots"],
    "crack": ["crack"],
    "hole": ["hole"],
    "burn": ["burn mark"],
    "peeling": ["peeling paint"],
    "other": ["damage"],
}


class DamageFinder:

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

    def find(self, segments: list, duration: float) -> dict:
        if not segments:
            return {"damages": []}

        prompt = DAMAGE_PROMPT.format(duration=duration, transcript=self.format_transcript(segments))
        messages = [SystemMessage(content=DAMAGE_SYSTEM_PROMPT), HumanMessage(content=prompt)]

        result = self.llm_client.run(self.model_name, messages, DamageMentionList)
        damages = self.clean_up([mention.model_dump() for mention in result.damages], duration)

        return {"damages": damages}

    def start_of(self, damage: dict) -> float:
        return damage["start_seconds"]

    def clean_search_terms(self, damage: dict) -> list:
        terms = []
        for term in damage["search_terms"] + DEFAULT_SEARCH_TERMS[damage["damage_class"]]:
            term = term.strip().lower()
            if term and term not in terms:
                terms.append(term)
        return terms[:MAX_SEARCH_TERMS]

    def clean_up(self, damages: list, duration: float) -> list:
        kept = []

        for damage in sorted(damages, key=self.start_of):
            start = min(max(damage["start_seconds"], 0.0), duration)
            end = min(max(damage["end_seconds"], start + MIN_DURATION_S), duration)
            damage["start_seconds"] = round(start, 2)
            damage["end_seconds"] = round(end, 2)
            damage["search_terms"] = self.clean_search_terms(damage)

            is_repeat = False
            for other in kept:
                same_class = other["damage_class"] == damage["damage_class"]
                if same_class and damage["start_seconds"] <= other["end_seconds"] + SAME_DAMAGE_GAP_S:
                    other["end_seconds"] = max(other["end_seconds"], damage["end_seconds"])
                    is_repeat = True

            if not is_repeat:
                kept.append(damage)

        for index, damage in enumerate(kept):
            damage["id"] = f"X{index + 1}"

        return kept
