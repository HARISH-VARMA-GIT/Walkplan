"""Walkplan floor plan schema (v0.1). Plain data models only.

Conventions
- Units: metres (areas in m^2).
- Plan frame: x to the right, y up.
- A room is a closed loop of walls in order: walls[i].end == walls[i+1].start.
- Wall coordinates are the inner face of the wall.
- Every measured number is a Measurement (value + low/high interval).

Checks and calculations (closing the loop, areas, intervals, save/load) live in walkplan/geometry.py.
"""
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

Point = tuple[float, float]


class Measurement(BaseModel):
    value: float
    low: float
    high: float
    method: Literal["measured", "estimated", "assumed"] = "measured"
    source: Optional[str] = None  # e.g. "lidar_plane_fit", "gpt-4o", "door_height_prior"


class Wall(BaseModel):
    id: str
    start: Point
    end: Point
    thickness_m: float = 0.12
    length: Optional[Measurement] = None  # if missing, use the distance start -> end


class Opening(BaseModel):
    id: str
    type: Literal["door", "window", "opening"]
    wall_id: str
    offset_m: float  # distance along the wall from wall.start to the opening's near edge
    width: Measurement
    height: Optional[Measurement] = None
    sill_height: Optional[Measurement] = None  # windows only
    leads_to: Optional[str] = None  # room id on the other side
    swing: Literal["left", "right", "none"] = "left"


class Damage(BaseModel):
    id: str
    surface_type: Literal["wall", "floor", "ceiling"]
    wall_id: Optional[str] = None  # set when surface_type == "wall"
    damage_class: Literal["water_stain", "mold", "crack", "hole", "burn", "peeling", "other"]
    extent_m2: Measurement
    confidence: float = Field(ge=0, le=1)
    concealed_flag: Optional[str] = None  # rule that fired if hidden damage is suspected
    notes: Optional[str] = None


class Room(BaseModel):
    id: str
    label: Literal["bedroom", "bathroom", "kitchen", "living_room", "dining_room", "hallway", "closet",
                   "laundry", "office", "garage", "stairs", "entry", "other"] = "other"
    name: Optional[str] = None
    level: int = 0
    walls: list[Wall]
    openings: list[Opening] = []
    ceiling_height: Optional[Measurement] = None
    floor_area: Optional[Measurement] = None
    damage: list[Damage] = []


class Adjacency(BaseModel):
    room_a: str
    room_b: str
    via_opening: Optional[str] = None


class CaptureInfo(BaseModel):
    id: str
    tier: Literal["photos", "video", "lidar"]
    device: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "0.1.0"


class FloorPlan(BaseModel):
    units: Literal["m",'cm','mm'] = "m"
    capture: CaptureInfo
    rooms: list[Room]
    adjacency: list[Adjacency] = []
    notes: str