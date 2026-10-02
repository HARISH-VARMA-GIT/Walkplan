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
