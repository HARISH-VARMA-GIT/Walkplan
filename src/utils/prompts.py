ROOM_IMAGES_SYSTEM_PROMPT = "You are an expert at reading room photos for building floor plans."

ROOM_IMAGES_PROMPT = """These photos show one room of a property. The images are given in the order listed below.

Image files in order:
{file_names}

Describe each image and then describe the whole room.
Do not guess what you cannot see. Use "unknown" when unsure."""

ROOM_LAYOUT_SYSTEM_PROMPT = "You are an expert at reading room photos to work out the layout of a floor plan."

ROOM_LAYOUT_PROMPT = """These {photo_count} photos show ONE room, taken from different spots inside it.
A 3D model has drawn overlays on each photo:
- Each coloured patch with a big letter (A, B, C ...) is a flat vertical surface found in that photo. Most are room walls, but some can be furniture fronts (wardrobe, headboard), mirrors or curtains.
- White numbered circles are corners where two of those surfaces meet.
- Yellow boxes are detected doors and cyan boxes are detected windows, labelled O1, O2 ... The full opening id is <photo id>-<label>, for example P03-O1.
- In every photo, walls from left to right follow the clockwise direction around the room.
- Letters are given per photo: wall A in one photo is usually a different wall than wall A in another photo. Match walls by what is on them (doors, windows, wardrobe, lights, switches) and by their neighbours.

Measurements from the 3D model (metres, for context only):
{photo_summaries}

Your job is only the layout. Do not estimate any lengths.
1. Work out the room outline seen from above. List every room wall once as W1, W2, ... going clockwise seen from above. Start W1 at the wall with the main entrance door. Include short walls such as the sides of an alcove or niche. For each wall say if the corner at its end is "normal" (usual inside corner) or "protruding" (sticks into the room). A closed room always has exactly 4 more normal corners than protruding corners (a plain rectangle has 4 normal and 0 protruding).
2. For every photo and every wall letter, give the room wall it shows, or "none" if it is furniture, a mirror, a curtain or not a wall.
3. For every opening id, give the physical door or window it is. Use the same id (D1, D2, N1 ...) when the same door or window appears in several photos. Give its true type and the room wall it is on. Use "none" for false detections.
Do not guess what you cannot see. Explain doubts in notes."""
