ROOM_IMAGES_SYSTEM_PROMPT = "You are an expert at reading room photos for building floor plans."

ROOM_IMAGES_PROMPT = """These photos show one room of a property. The images are given in the order listed below.

Image files in order:
{file_names}

Describe each image and then describe the whole room.
Do not guess what you cannot see. Use "unknown" when unsure."""
