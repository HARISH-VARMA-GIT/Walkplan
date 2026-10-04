# Photo tier: room photos → floor plan

This document explains how Walkplan turns a folder of room photos into a floor plan (`FloorPlan` JSON + SVG), how to run it, how to read the outputs, and what does not work yet.

> **Status (2026-10-03, branch `features/map-anything`):** the pipeline runs end to end on `src/inputs/images/room1` (8-photo centre sweep). Camera poses come from **MapAnything** and are reliable: the 8 headings come out in clean 45° steps. Room1 traces as an 8-wall outline, 3.32 m wide at the top and 4.65 m at the bottom, area about 18.3 m². **Not yet checked against a tape measure.**

---

## 1. The idea in one paragraph

A single photo has no depth and no camera position. **MapAnything** looks at all photos together and gives every photo its camera position and rotation in one shared 3D frame. **MoGe-2** turns every photo into an accurate 3D point map in metres. From that we find the floor and walls *in each photo* and measure them: distance from the camera to each wall, door and window sizes. Because all photos share one frame, walls from different photos can be lined up directly, and the room outline falls out of the geometry. No LLM guessing is needed. The LLM layout path is still there as an option (`--layout llm`).

```
photos
  │
  ├─ 1. poses        MapAnything (all photos at once) → camera rotation + position per photo
  ├─ 2. depth        MoGe-2 per photo → 3D points in metres
  ├─ 3. heights      shared "up" from MapAnything, floor height shared between photos
  ├─ 4. openings     Grounding DINO ("door. window.") + SAM → pixel masks
  ├─ 5. measure      per photo: RANSAC floor/walls, wall distances, corners,
  │                  door/window width + height (rays cast onto the wall plane)
  ├─ 6. place        put every photo in one room frame (scale, gravity, walls aligned to x/y)
  ├─ 7. layout       cluster walls from all photos → carve free space → trace outline
  ├─ 8. assemble     least-squares wall positions → walls, openings, ceiling
  └─ 9. render       FloorPlan JSON → SVG
```

---

## 2. Setup

One virtual environment at the repo root (`.venv`, Python 3.11) holds everything, including MoGe and MapAnything.

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run this from the repo root (`requirements.txt` installs MoGe with `-e ./src/MoGe`). It pulls CUDA PyTorch (`cu130`, works on RTX 50xx) and MapAnything from GitHub. MapAnything needs `opencv-python-headless==4.10.0.84`. Do not also install `opencv-python`; both provide `cv2` and clash.

`.env` (copy from `.env.example`):

| Key | Used for |
|---|---|
| `OPENAI_API_KEY` | only for `--layout llm` and the old photo-description step |
| `MAPANYTHING_MODEL_NAME` | default `facebook/map-anything-apache` (Apache 2.0, commercial use OK). `facebook/map-anything` is CC-BY-NC (research only) |
| `LAYOUT_MODEL_NAME` | model for `--layout llm` (default `gpt-5.5`) |
| `HF_HOME`, `TORCH_HOME` | model cache folders. Default: `<repo>/.cache/` (set in `src/app.py`, gitignored) |

Models download on first run, about 7 GB in total: MapAnything 4.7 GB plus its DINOv2 code, MoGe 1.3 GB, DINO 0.7 GB, SAM 0.4 GB. If a download fails with a memory error, set `HF_HUB_DISABLE_XET=1`.

---

## 3. Taking photos (centre sweep)

1. Stand near the middle of the room, on open floor.
2. **Photo 1 faces the entrance door.** Then turn **clockwise** about 45° per photo, 8 photos in total.
3. Turn on the spot (feet in place, phone near your chest), phone level, **landscape, 0.5× ultra-wide**, same lens for all.
4. Name the files in order: `01.jpg … 08.jpg` (camera names like `IMG_…120212.jpg` also work, since they sort by time).
5. Send original files (not WhatsApp).
6. Optional: tilt slightly up so some ceiling is visible, which gives ceiling height. Tape-measure two walls and one door for checking.

---

## 4. Running it

```
.venv\Scripts\python.exe src\app.py --images src\inputs\images\room1 --room-type bedroom
```

| Option | Meaning |
|---|---|
| `--output <folder>` | where results go (default `output/<room name>`) |
| `--room-type` | label in the plan (`bedroom`, `kitchen`, …) |
| `--layout geometry` / `llm` | `geometry` (default): outline from MapAnything poses. `llm`: old vision-LLM path |
| `--redo-photos` | measure the photos again (stages 3–5). Depth and poses stay cached |
| `--redo-layout` | rebuild the layout (stage 7) |

Caching: poses and depth are cached per photo and keyed by file name, so changing photos recomputes them. Other stages reuse their JSON unless you pass `--redo-…`. You can hand-edit `layout.json` and re-run without flags (only assembly and SVG run, about 1 s).

---

## 5. Stage by stage (where the code lives)

All code is in `src/modules/image_processing/` unless noted. The orchestration is `main.py` `ImageProcessingService.build_floor_plan`.

| Stage | File / class | What it does | Output |
|---|---|---|---|
| 1 poses | `pose_estimator.py` `PoseEstimator` | MapAnything on all photos together → cam-to-world pose (OpenCV axes) and MapAnything depth. Unloaded afterwards to free GPU memory | `poses/P01.npz` … |
| 2 depth | `depth_estimator.py` `DepthEstimator` | MoGe-2 → points (x right, y down, z forward, metres), mask, intrinsics | `depth/P01.npz` … |
| 3 heights | `main.py` `height_hints` + `photo_placer.py` | World "up" from the cameras (refined by floor planes that agree). Each photo gets its own up vector. Photos that see the floor give the floor level, and the others get their camera height from the MapAnything poses | log line "Camera heights from shared floor" |
| 4 openings | `opening_finder.py` `OpeningFinder` | Grounding DINO boxes (duplicates and boxes over 40% of the image dropped) → SAM masks | – |
| 5 measure | `plane_finder.py`, `photo_measurer.py` `PhotoMeasurer`, `opening_finder.py` `OpeningMeasurer` | RANSAC planes with door/window pixels removed. Floor/walls classified using the up hint. Walls get letters, distances, visible extents, corners. Openings are measured by casting rays onto their wall plane | `photo_geometry.json`, `overlays/*.jpg` |
| 6 place | `photo_placer.py` `PhotoPlacer` | MapAnything→MoGe scale (median depth ratio). Every photo's heading and position in a gravity-aligned frame rotated so walls run along x/y | `placements.json` |
| 7 layout | `geometry_layout.py` `GeometryLayoutBuilder` | See section 6 | `layout.json` |
| 8 assemble | `room_assembler.py` `RoomAssembler`, `wall_position_solver.py` `WallPositionSolver` | See section 6 | `wall_matches_used.json`, `floor_plan.json` |
| 9 render | `src/modules/floor_plan_generator/` `FloorPlanRenderer`, `SvgCreator` | SVG with walls, door swings, windows, dimensions ±, opening labels (`D1 0.68 × 2.14 m`), area and ceiling | `floor_plan.svg` |

Schemas: `src/models/photo_geometry.py` (per-photo results, placements), `src/models/image_models.py` (`RoomLayout`), `src/models/floor_plan.py` (final plan). Plan helper maths is in `src/utils/geometry.py`.

---

## 6. How the outline is found (stages 7–8)

1. **World walls.** Every photo wall (≥ 0.3 m visible) is moved into the room frame using its photo's heading and position, then snapped to one of 4 directions (facing east, west, north or south).
2. **Wall lines.** Walls with the same direction and position (within 25 cm) are clustered. Each cluster is one wall line.
3. **Grid.** Grid lines are drawn at every wall line, and also where long clusters end. That is how steps and alcoves get their own cells even when their short side face was never photographed.
4. **Free-space carving.** For every visible wall point, the line of sight from the camera to it (stopping 30 cm short) marks the grid cells it crosses as "inside". Cells with few hits compared to the typical cell are ignored (noise).
5. **Outline.** Keep the biggest inside region, fill holes, trace its border clockwise, and merge straight runs into walls W1…Wn. W1 is the top wall. Each corner is `normal` (right turn) or `protruding` (left turn).
6. **Clean-up.** A short wall (≤ 0.6 m) is removed when that changes the floor area by less than 0.3 m², either by moving a neighbour wall onto the next line (a step) or by filling a notch. Noisy poses cause these slivers; real steps like room1's 0.43 m one (about 0.7 m²) stay.
7. **Matching.** Each photo wall is matched to the outline wall with the same direction, close position and overlapping extent. Doors and windows seen in several photos are merged by position along their wall (within 0.5 m).
8. **Solve.** `WallPositionSolver` finds every wall position and camera position by least squares. Inputs: camera-to-wall distances, MapAnything headings (fixed), MapAnything positions (soft, ±0.3 m), outline positions (weak, ±0.5 m, only matters for walls nobody saw), and "visible wall ends must be inside the room" (only at normal corners). Outliers are dropped. Length ± comes from the solver; `method` is `measured` if the wall and both neighbours were seen.

Tested on a synthetic L-shaped room with 6 walls (5, 2, 2, 2, 3, 4 m): with up to 10 cm noise on camera positions every wall comes back within about 0.1 m. With 20–30 cm noise the shape is still 6 walls, but lengths can be off by 0.5 m or more.

---

## 7. Reading the outputs

| File | Look at |
|---|---|
| `overlays/*.jpg` | Walls (colour + letter), corners, door/window boxes per photo |
| `placements.json` | `heading_deg` per photo (centre sweep: steps of about 45°), `position`, `scale` |
| `photo_geometry.json` | wall `distance_from_camera_m`, `visible_length_m`; opening `width_m`, `height_m`, `looks_valid`; `camera_height_m` |
| `layout.json` | Outline walls (`corner_after`, `position_m`) and matches. **Editable** |
| `wall_matches_used.json` | Matches the solver used |
| `floor_plan.json` | Final `FloorPlan`. Every number is a `Measurement` (`value`, `low`, `high`, `method`) |
| `floor_plan.svg` | Open in a browser |
| console warnings | Unseen walls, odd door sizes, ignored matches. Also in `floor_plan.json` `notes` |

---

## 8. Known limitations

- **Not checked against ground truth yet.** Tape-measure room1 and compare.
- **Ceiling height:** missing when the ceiling isn't in frame (level landscape photos rarely show it).
- **Door detections are noisy:** wardrobe doors, curtains and glass doors become extra "doors" (room1 currently lists several on one wall). Most are flagged `size outside normal range`. Needs better filtering, or the LLM to label them.
- **Step/alcove side faces** that no photo looks at are placed from the outline only (`estimated`, wider ±).
- **Mirrors:** Grounding DINO is asked for "door. window. mirror.". Mirror pixels are left out of the wall planes, so their fake depth no longer pushes a wall out, and mirrors are never measured as openings. A mirror DINO calls a window is still missed.
- Manhattan rooms only (90° corners). Multi-room stitching only for the video tier (`docs/video-pipeline.md` section 7). Damage only from video.
- GPU: needs about 8 GB. Each model is unloaded before the next one loads.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `There is not enough space on the disk` while downloading | model cache defaults to `<repo>/.cache`. Make sure that drive has about 10 GB free |
| `MemoryError` / `memory allocation … failed` while downloading | `set HF_HUB_DISABLE_XET=1` and run again (downloads resume) |
| `cannot import name 'httpx' from 'huggingface_hub.utils'` | `transformers` 5 needs `huggingface_hub>=1.33`: re-run `pip install -r requirements.txt` |
| `cv2` errors after installing something | both `opencv-python` and `opencv-python-headless` installed. Uninstall both, then `pip install opencv-python-headless==4.10.0.84` |
| `CUDA out of memory` | close other GPU programs. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| `Not enough walls found in two directions` | photos don't show walls on two axes. Do the full 8-photo sweep |
| Outline looks wrong | look at `overlays/` (are letters on real walls?) and `placements.json` (headings in 45° steps?). Fix `layout.json` and re-run |
| `uv run` inside `src/MoGe` undoes upgrades | use the root `.venv`, not `uv run` |

---

## 10. Other scripts

`src/MoGe/measure_distance.py` (click two points, get a distance) and `src/MoGe/measure_distance_with_sam.py` (auto-measure one object by name) are standalone experiments.
