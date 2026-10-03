from typing import Literal, Optional

from pydantic import BaseModel


Point2D = tuple[float, float]
EndType = Literal["corner", "truncated", "occluded"]


class PhotoWall(BaseModel):
    letter: str
    start: Point2D
    end: Point2D
    visible_length_m: float
    start_type: EndType
    end_type: EndType
    distance_from_camera_m: float
    visible_height_m: float
    image_center_x: float
    point_count: int


class PhotoCorner(BaseModel):
    number: int
    wall_before: str
    wall_after: str
    point: Point2D
    angle_deg: float
    corner_type: Literal["normal", "protruding"]


class PhotoOpening(BaseModel):
    id: str
    type: Literal["door", "window"]
    wall_letter: Optional[str] = None
    box: list[float]
    score: float
    start_along_wall_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    sill_height_m: Optional[float] = None
    looks_valid: bool = False


class PhotoGeometry(BaseModel):
    photo_id: str
    image_path: str
    image_width: int
    image_height: int
    fov_x_deg: float
    floor_found: bool
    camera_height_m: Optional[float] = None
    ceiling_height_m: Optional[float] = None
    walls: list[PhotoWall] = []
    corners: list[PhotoCorner] = []
    openings: list[PhotoOpening] = []
    notes: list[str] = []


class RoomPhotoGeometry(BaseModel):
    room_name: str
    photos: list[PhotoGeometry]
