import os

from openai import OpenAI


DEFAULT_TRANSCRIBE_MODEL = "whisper-1"


class Transcriber:

    def __init__(self):
        self.model_name = os.getenv("TRANSCRIBE_MODEL_NAME") or DEFAULT_TRANSCRIBE_MODEL
        self.client = None

    def transcribe(self, audio_path: str) -> dict:
        if self.client is None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open(audio_path, "rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
            )

        data = result.model_dump()

        segments = []
        for segment in data.get("segments") or []:
            text = segment["text"].strip()
            if text:
                segments.append({"start_seconds": round(segment["start"], 2), "end_seconds": round(segment["end"], 2), "text": text})

        words = []
        for word in data.get("words") or []:
            words.append({"start_seconds": round(word["start"], 2), "end_seconds": round(word["end"], 2), "word": word["word"]})

        return {
            "model": self.model_name,
            "language": data.get("language"),
            "text": data.get("text", ""),
            "segments": segments,
            "words": words,
        }
