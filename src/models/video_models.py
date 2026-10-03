from typing import Literal

from pydantic import BaseModel, Field


DamageClass = Literal["water_stain", "mold", "crack", "hole", "burn", "peeling", "other"]


class DamageMention(BaseModel):
    start_seconds: float = Field(description="Start time of the transcript part where this damage is talked about")
    end_seconds: float = Field(description="End time of the transcript part where this damage is talked about")
    damage_class: DamageClass = Field(description="Type of damage")
    surface_type: Literal["wall", "floor", "ceiling", "unknown"] = Field(description="Surface the damage is on")
    location: str = Field(description="Where the speaker says it is, e.g. 'left wall below the window'. Empty if not said")
    description: str = Field(description="Short damage description in English, e.g. 'brown water stain about 30 cm wide'")
    severity: Literal["minor", "moderate", "severe", "unknown"] = Field(description="Severity if the speaker says it or it is clear")
    quote: str = Field(description="The speaker's words about this damage, copied from the transcript")
    search_terms: list[str] = Field(description="1 to 3 short English visual names for an object detector, e.g. ['water stain', 'stain']")


class DamageMentionList(BaseModel):
    damages: list[DamageMention] = Field(description="Every separate damage the speaker talks about. Empty list if none")


RoomLabel = Literal["bedroom", "bathroom", "kitchen", "living_room", "dining_room", "hallway", "closet",
                    "laundry", "office", "garage", "stairs", "entry", "other"]


class RoomMention(BaseModel):
    start_seconds: float = Field(description="Time the speaker starts showing this room (0 for the first room)")
    name: str = Field(description="Room name as the speaker calls it, e.g. 'master bedroom', 'room 2'")
    room_type: RoomLabel = Field(description="Type of room")


class RoomMentionList(BaseModel):
    rooms: list[RoomMention] = Field(description="Rooms in the order they are filmed. One item if the video shows one room")


class VideoFrame(BaseModel):
    photo_id: str
    file_name: str
    time_seconds: float
    sharpness: float
    purpose: Literal["layout", "damage", "both", "transition"]
    damage_ids: list[str] = []


class DamageBoxChoice(BaseModel):
    box_number: int = Field(description="Number of the box that shows the damage, or 0 if no box shows it")
    grid_cell: Literal["top-left", "top-center", "top-right", "middle-left", "center", "middle-right",
                       "bottom-left", "bottom-center", "bottom-right", "not visible"] = Field(
        description="Part of the image (3 x 3 grid) where the damage is, or 'not visible' if it is not in the image")
    reason: str = Field(description="One short sentence on what you see")
