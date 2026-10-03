from typing import Literal, Optional

from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    file_name: str = Field(description="Name of the image file")
    description: str = Field(description="Short description of what is visible")
    visible_walls: str = Field(description="Which walls, corners or doors can be seen")
    doors_and_windows: str = Field(description="Number and position of doors and windows")
    fixtures_and_furniture: list[str] = Field(description="List of main items")
    visible_damage: str = Field(description="Any damage seen, or none")


class RoomImageAnalysis(BaseModel):
    room_type: str = Field(description="bedroom, kitchen, bathroom, living room, hallway, etc.")
    images: list[ImageDescription]
    overall_description: str = Field(description="Short description of the whole room")
    estimated_shape: str = Field(description="rectangle, L shape, or unknown")


class RoomWallInfo(BaseModel):
    id: str = Field(description="Room wall id: W1, W2, ... in clockwise order seen from above")
    description: str = Field(description="What is on or near this wall, e.g. 'entrance door and wardrobe'")
    corner_after: Literal["normal", "protruding"] = Field(
        description="Corner at the end of this wall (going clockwise). 'normal' = usual inside corner of the room. "
                    "'protruding' = a corner that sticks into the room, like the edge of an alcove or a pillar"
    )
    position_m: Optional[float] = Field(default=None, description="Leave empty")


class WallMatch(BaseModel):
    photo_id: str = Field(description="Photo id exactly as given, e.g. P03")
    wall_letter: str = Field(description="Wall letter drawn on that photo")
    room_wall_id: str = Field(description="Matching room wall id, e.g. W2, or 'none' if it is not a room wall (furniture, mirror)")


class OpeningMatch(BaseModel):
    photo_opening_id: str = Field(description="Opening id drawn on a photo, e.g. 'P03-O1'")
    room_opening_id: str = Field(description="Same id for the same physical door or window across photos, e.g. D1, D2, N1. 'none' if it is a false detection")
    type: Literal["door", "window", "none"] = Field(description="What it really is")
    room_wall_id: str = Field(description="Room wall it sits on, e.g. W3")


class RoomLayout(BaseModel):
    room_type: Literal["bedroom", "bathroom", "kitchen", "living_room", "dining_room", "hallway", "closet",
                       "laundry", "office", "garage", "stairs", "entry", "other"]
    shape_description: str = Field(description="Short description of the room outline, e.g. 'rectangle with a small alcove on W3'")
    walls: list[RoomWallInfo] = Field(description="All room walls in clockwise order, starting with the wall that has the main entrance door")
    wall_matches: list[WallMatch]
    opening_matches: list[OpeningMatch]
    notes: str = Field(description="Anything uncertain")
