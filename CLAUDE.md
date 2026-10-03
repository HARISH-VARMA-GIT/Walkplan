# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Walkplan: generate floor plans from three input tiers — photos (2–8 stills per room, per-room folders, must stitch into whole-property plan), video (handheld iPhone 15+ walkthrough), LiDAR (depth + poses + intrinsics, Pro-class devices). Target output is Xactimate-compatible `.esx` (proprietary Verisk format; reverse-engineered from `esx sample downloadable.esx`). See `docs/spec.md` (requirements, test set, benchmark = LiDAR vs best competitor app), `docs/theory.md` (Xactimate/Magicplan/Encircle background), `docs/TODO.md`.

Photo tier works end to end with MapAnything camera poses; not yet validated against tape measurements — see `docs/photo-pipeline.md`. Video tier (branch `features/video-processing`) reuses the photo pipeline on frames picked from the video and places spoken damage on the plan — see `docs/video-pipeline.md`. LiDAR `main.py` is a stub.

## Setup / Commands

Windows, Python 3.11, single venv in `.venv/` at repo root (also holds MoGe; `src/MoGe/.venv` is obsolete). Deps: `.venv\Scripts\python.exe -m pip install -r requirements.txt` from repo root (installs torch cu130, transformers 5, `-e ./src/MoGe`, MapAnything from git, `opencv-python-headless==4.10.0.84` — never also install `opencv-python`). ffmpeg on PATH (or `FFMPEG_PATH`, fallback `imageio-ffmpeg`). Model caches default to `<repo>/.cache/` (`HF_HOME`/`TORCH_HOME` set in `src/app.py`; C: drive is full). Copy `.env.example` to `.env`. No tests, linter or build configured yet.

Photo pipeline (outputs to `output/<room>/`; stages cached, edit `layout.json` and re-run to re-assemble only):

```
.venv\Scripts\python.exe src\app.py --images src\inputs\images\room1 --room-type bedroom [--redo-photos] [--redo-layout] [--layout geometry|llm]
```

Video pipeline (one video per room folder; outputs to `output/<room>_video/`; transcript, damage list and frames cached):

```
.venv\Scripts\python.exe srcpp.py --video src\inputsideosoom1 --room-type bedroom [--max-frames 24] [--redo-video] [--redo-photos] [--redo-layout]
```

LiDAR scripts (Stray Scanner capture folder: `rgb.mp4`, `depth/`, `confidence/`, `camera_matrix.csv`, `odometry.csv`), run from `src/modules/lidar_processing/`:

```
python stray_to_3d.py ../../../data/single_room/<capture> --every 3   # -> mesh.ply, points.ply (--no-flip if mesh smeared)
python measure_room.py ../../../data/single_room/<capture>            # -> room.json, room_plan.png (needs points.ply)
```

`data/` (sample captures, zips) is gitignored.

## Architecture

- `src/modules/<input>_processing/` — one high-level feature module per input tier.
- `src/models/` — all pydantic schemas.
- `src/utils/` — shared helpers.
- LiDAR pipeline: `stray_to_3d.py` fuses depth + odometry into point cloud/mesh; `measure_room.py` works in ARKit world (y-up, gravity-aligned): floor/ceiling via height histogram, rotate to axis-aligned walls, wall lines from 1-D peaks, grid-cell room shape (handles L-shapes), polygon -> wall lengths + area.
- Photo pipeline (`src/modules/image_processing/`): MapAnything poses (`pose_estimator.py`) → MoGe-2 depth → shared up/floor-height hints → Grounding DINO + SAM openings → RANSAC planes, per-photo wall distances/corners (`photo_geometry.json`) → `photo_placer.py` puts photos in one gravity/wall-aligned frame (`placements.json`) → `geometry_layout.py` clusters walls, carves free space on a grid, traces outline, removes small jogs (`layout.json`; `--layout llm` = old LLM path) → `WallPositionSolver` least squares (fixed headings, position priors) → `FloorPlan` → SVG (`src/modules/floor_plan_generator/`, helpers in `src/utils/geometry.py`).
- Video pipeline (`src/modules/video_processing/`): ffmpeg audio → OpenAI `whisper-1` timed transcript → LLM damage mentions (`damage_finder.py`, ids X1..) → 2 fps candidate frames, sharpest per time slot + 2 per damage mention (`frame_selector.py`) → `ImageProcessingService.build_floor_plan_from_paths(..., damage_requests, damage_details)` → shared FOV from MapAnything fed to MoGe (one lens per video) → `DamageMeasurer` (DINO candidate boxes, vision LLM picks one in `damage_box_picker.py`, SAM, ray cast to wall/floor) → `RoomAssembler.build_damages` (wall + offset + height via solver pose) → SVG damage markers + legend. `PoseEstimator` loads MapAnything on the meta device and streams weights to GPU (Windows commit memory is tight).
- Floor plan schema: `src/models/floor_plan.py` (metres, x right / y up, walls clockwise or ccw closed loop, every number a `Measurement` with low/high).
- Planned: validate vs tape; better door filtering / mirror masking; better damage detection; multi-room stitching; `.esx` export.

## Code Style (from `docs/code-instructions.md`)

- Always OOP.
- Beginner-friendly: simple, readable code; beginner-friendly class/function/variable names.
- Avoid heavy comments. Do not write multi line comments
- Use spaces/whitespace generously.
- Keep OOP simple: plain classes, `__init__`, instance methods, basic inheritance. Avoid `@staticmethod`, `@classmethod`, `@contextmanager`, `@abstractmethod`/ABC, `@property`, and similar decorators. Base class stubs raise `NotImplementedError`.
- No lambdas; use normal named functions/methods.
- Functions returning LLM output return a plain dict keyed by name, e.g. `{"folder_path": folder_path, "analysis": analysis.model_dump()}`.

## LLM Access

`src/utils/llm_provider.py` is the single entry point for LLM calls. `LlmProviderFactory` picks the provider from `LLM_PROVIDER` (default `openai`); `LlmClient.run(llm_model_name, messages, output_model)` takes an LLM model name (pass `None` to use the fallback chain), messages and a pydantic schema, returns the structured result, and handles retry + model fallback using the `LLM_MODEL_CHAIN` env var; `VisionLimiter` caps concurrent vision calls (`LLM_VISION_MAX_CONCURRENT`). Add new providers as `LlmProvider` subclasses registered in the factory.
