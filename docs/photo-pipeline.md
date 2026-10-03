# Photo tier: room photos → floor plan

This document explains how Walkplan turns a folder of room photos into a floor plan (`FloorPlan` JSON + SVG), how to run it, how to read the outputs, and what does not work yet.

> **Status (2026-10-03):** the pipeline runs end to end on `src/inputs/images/room1`, but the room size is **not reliable yet**. Per-photo measurements are good; deciding which way each photo faces is not (see [Known limitations](#known-limitations)). Next step: add MapAnything for camera poses on a separate branch.

---

## 1. The idea in one paragraph

A single photo has no depth and no camera position. **MoGe-2** turns every photo into a 3D point map in metres. From that we find the floor, the ceiling and the walls *in that photo*, and measure them: how far each wall is from the camera, where the corners are, and how big the doors and windows are. Then all photos are combined into one room. Numbers come from geometry. The vision LLM is only asked about *topology* ("which wall is this?"), because LLM-guessed lengths were wrong.

```
photos
  │
  ├─ 1. depth        MoGe-2 per photo → 3D points (metres)
  ├─ 2. openings     Grounding DINO ("door. window.") + SAM → pixel masks
  ├─ 3. planes       RANSAC → floor / ceiling / walls (door pixels ignored)
  ├─ 4. measure      per photo: wall distances, visible lengths, corners, ceiling height,
  │                  door/window width + height (rays cast onto the wall plane)
  ├─ 5. layout       overlays → LLM: room walls W1..Wn clockwise + which photo wall is which
  ├─ 6. assemble     repair LLM matches with geometry → solve wall positions + camera positions
  │                  → walls, openings, ceiling
  └─ 7. render       FloorPlan JSON → SVG
```

---

## 2. Setup

One virtual environment at the repo root (`.venv`, Python 3.11) holds everything, including MoGe.

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run this from the repo root, because `requirements.txt` installs MoGe with `-e ./src/MoGe`. It also pulls a CUDA build of PyTorch (`cu130`, works on RTX 50xx) through `--extra-index-url`.

`.env` (copy from `.env.example`):

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | LLM calls |
| `LAYOUT_MODEL_NAME` | model for the layout step. Default chain: `gpt-5.5`, then `gpt-4.1` (`LLM_LAYOUT_MODEL_CHAIN`) |
| `VISION_MODEL_NAME` | old photo-description step (`process_image`) |

Models download from Hugging Face on first run and are cached in `~\.cache\huggingface\hub`:
`Ruicheng/moge-2-vitl-normal`, `IDEA-Research/grounding-dino-tiny`, `facebook/sam-vit-base`.

---

## 3. Running it

```
.venv\Scripts\python.exe src\app.py --images src\inputs\images\room1
```

| Option | Meaning |
|---|---|
| `--output <folder>` | where results go (default `output/<room name>`) |
| `--redo-photos` | measure the photos again (stages 2–4). Depth stays cached |
| `--redo-layout` | ask the LLM for the layout again (stage 5) |

Every stage saves its result, and a plain re-run reuses them. So you can:

- **fix the layout by hand**: edit `layout.json`, then re-run without flags. Only assembly and SVG run again (no GPU, no LLM, about 1 second).
- **re-measure after changing measurement code**: `--redo-photos`. MoGe is skipped because depth is cached, and the layout is asked again.
- **start completely fresh**: delete the output folder.

If the GPU runs out of memory, close other GPU programs, or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

---

## 4. Stage by stage (where the code lives)

All code is in `src/modules/image_processing/` unless noted.

| Stage | File / class | What it does | Output |
|---|---|---|---|
| 1 depth | `depth_estimator.py` `DepthEstimator` | MoGe-2 → points (x right, y down, z forward, metres), valid mask, normals, intrinsics. Model is unloaded after all photos to free GPU memory | `depth/P01.npz` … |
| 2 openings | `opening_finder.py` `OpeningFinder` | Grounding DINO finds all door/window boxes (duplicates removed), SAM turns each box into a mask | – |
| 3 planes | `plane_finder.py` `PlaneFinder`, `SurfaceClassifier` | Sequential RANSAC (open3d, 3 cm). Floor = lowest upward-facing plane, which also gives true "up". Ceiling = highest downward-facing plane. Walls = vertical planes. Door/window pixels are removed first, so door leaves and wardrobe fronts don't become walls | – |
| 4 measure | `photo_measurer.py` `PhotoMeasurer` | Puts each photo in a top-down frame (camera at 0,0, looking along +y). Merges near-duplicate planes (< 15 cm, < 6°), drops short planes (furniture). Each wall gets a letter left→right, a visible start/end, a distance from the camera, and end types `corner` / `truncated` (leaves the photo) / `occluded`. Corners are where two walls meet, one corner per wall end. Ceiling height = floor-to-ceiling plane distance | `photo_geometry.json` |
| 4 openings | `opening_finder.py` `OpeningMeasurer` | Picks the wall the opening sits on, then casts camera rays through the mask onto that wall plane. This avoids bad depth on glass and mirrors. Gives width, height, sill and position along the wall, plus a `looks_valid` flag (door 0.6–1.3 m wide, 1.8–2.6 m high) | in `photo_geometry.json` |
| overlays | `overlay_drawer.py` `OverlayDrawer` | Draws walls (colour + letter), corners (numbered circles), openings (yellow door / cyan window boxes) | `overlays/P01.jpg` … |
| 5 layout | `layout_reasoner.py` `LayoutReasoner` | Sends all overlays plus a text summary of the measurements to the LLM. It returns `RoomLayout` (`src/models/image_models.py`): room walls W1..Wn clockwise with `normal`/`protruding` corners, photo-wall → room-wall matches, and opening identities | `layout.json` |
| 6 assemble | `room_assembler.py` `RoomAssembler` + `wall_position_solver.py` `WallPositionSolver` | See section 5 | `wall_matches_used.json`, `floor_plan.json` |
| 7 render | `src/modules/floor_plan_generator/` `FloorPlanRenderer`, `SvgCreator` | FloorPlan → SVG with walls, door swings, windows, dimensions with ±, area and ceiling label | `floor_plan.svg` |

Schemas: `src/models/photo_geometry.py` (per-photo results), `src/models/image_models.py` (LLM output), `src/models/floor_plan.py` (final plan). Plan helper maths (polygon, area, intervals) is in `src/utils/geometry.py` `FloorPlanGeometry`.

---

## 5. How the room is assembled (stage 6)

1. **Wall directions.** Walk W1..Wn starting east. Turn right at a `normal` corner and left at a `protruding` one. A closed room needs exactly 4 more right turns than left turns, otherwise assembly stops and asks you to fix `layout.json`.
2. **Repair the LLM matches.** For each photo, the rotation that most LLM matches agree on (weighted by visible wall length) is taken as the photo's heading. Then every wall in that photo is re-assigned by its direction: in a rectangle each direction has exactly one wall. The LLM choice is kept only when several parallel walls fit, such as alcove sides.
3. **Solve.** Unknowns: one position per room wall, and one (x, y) per photo. Every wall seen in a photo gives one linear equation:
   `normal_of_wall · camera_position − wall_position = measured distance`.
   This is solved with weighted least squares (far walls count less). Outliers are dropped. Visible wall ends must lie inside the room, which works like a lower bound on length. Photos need two non-parallel walls to be placed.
4. **Walls** are the lines between consecutive corners, so the loop always closes. Length ± comes from the solver's covariance (at least 5%). `method` is `measured` when the wall and both neighbours were seen, otherwise `estimated`.
5. **Openings**: width and height are medians of the valid photo measurements. The position along the wall comes from the camera position, so no visible corner is needed.
6. **Ceiling**: median of the photos that saw both floor and ceiling.

The room size along one axis is only known if some photo sees **both opposite walls of that axis**, for example a wide photo showing the left and right walls. If no photo does, that length is only a lower bound, and the log says "some walls are not tied to any photo".

---

## 6. Reading the outputs

| File | Look at |
|---|---|
| `overlays/*.jpg` | Are letters on real walls? Doors and windows boxed? This is exactly what the LLM saw |
| `photo_geometry.json` | `distance_from_camera_m`, `visible_length_m`, `start_type`/`end_type`, door `width_m`/`height_m`, `looks_valid` |
| `layout.json` | LLM answer. **Editable**: change matches or corner types and re-run |
| `wall_matches_used.json` | Matches after geometry repair (what the solver actually used) |
| `floor_plan.json` | Final `FloorPlan`. Every number is a `Measurement` (`value`, `low`, `high`, `method`: `measured` / `estimated`) |
| `floor_plan.svg` | Open in a browser. Hover elements for tooltips |
| console warnings | Ignored matches, outliers, unseen walls, odd door sizes. Also copied into `floor_plan.json` `notes` |

---

## 7. Known limitations

**Photo orientation (main problem).** The solver needs to know which way each photo faces. That currently comes from the LLM's wall matching. Even `gpt-5.5` tends to label walls W1, W2, W3 in photo order. Geometry repair fixes 90° mistakes, but a photo rotated by 180° looks equally valid in a rectangular room. On room1 this gives about 2.4 × 1.8 m, while single photos clearly show about 3.3 m width.

- Feature matching (SIFT) links only 7 of 12 room1 photos: white walls plus WhatsApp compression leave little texture.
- **Planned fix:** add a multi-view model (MapAnything) only for camera poses, and keep MoGe scale, openings, the solver and the SVG.

**Other limitations**
- WhatsApp photos are 720×1280 with no EXIF, so MoGe has to guess the field of view. Expect a few percent scale error per photo.
- Portrait photos (~45° field of view) rarely show a whole wall.
- Ceiling height varies between photos (2.9–4.1 m on room1) because of the sloped ceiling. The median is reported.
- Wardrobes are often detected as "door". They are removed from wall finding (good), but the LLM has to mark them `none`.
- Manhattan walls only (90° corners). Single room only, no multi-room stitching yet. No damage detection yet.

---

## 8. Photo tips (for now)

- Send **original files** (not WhatsApp) to keep full resolution and focal length.
- Prefer **landscape**, stand back, and include floor and ceiling.
- Make photos overlap. Include views that show **two opposite walls at once** (stand in a corner and aim along the room).
- Capture every door and window fully (not cut off by the frame edge).

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `cannot import name 'httpx' from 'huggingface_hub.utils'` | `transformers` 5 needs `huggingface_hub>=1.33`. Re-run `pip install -r requirements.txt` |
| `huggingface-hub>=0.34.0,<1.0 is required` | old `transformers` 4.x installed. Upgrade: `pip install -U transformers huggingface_hub` |
| Using `uv run` inside `src/MoGe` undoes upgrades | `uv run` syncs to MoGe's `uv.lock`. Use the root `.venv` instead, or `uv run --no-sync` |
| `CUDA out of memory` | close other GPU programs. The pipeline already unloads MoGe before DINO/SAM |
| `Layout does not close a loop` | fix `corner_after` in `layout.json` (normal − protruding must be 4) and re-run |
| Room size clearly wrong | check `wall_matches_used.json` against `overlays/`, fix `layout.json` matches, re-run. See limitations |
| A door is missing | look at its overlay: no yellow box means DINO missed it. `looks_valid: false` means the size was out of range or it was cut off by the frame |

---

## 10. Other scripts

`src/MoGe/measure_distance.py` (click two points, get a distance) and `src/MoGe/measure_distance_with_sam.py` (auto-measure one object by name) are standalone experiments. The pipeline has its own copies of the depth and detection code in `src/modules/image_processing/`.
