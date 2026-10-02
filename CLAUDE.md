# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Walkplan: generate floor plans from three input tiers — photos (2–8 stills per room, per-room folders, must stitch into whole-property plan), video (handheld iPhone 15+ walkthrough), LiDAR (depth + poses + intrinsics, Pro-class devices). Target output is Xactimate-compatible `.esx` (proprietary Verisk format; reverse-engineered from `esx sample downloadable.esx`). See `docs/spec.md` (requirements, test set, benchmark = LiDAR vs best competitor app), `docs/theory.md` (Xactimate/Magicplan/Encircle background), `docs/TODO.md`.

Early stage: `src/modules/{image,video}_processing/main.py` and `lidar_processing/main.py` are empty stubs; `src/app.py` only loads `.env` and sets up logging.

## Setup / Commands

Windows, Python 3.11, venv in `.venv/`. Deps: `pip install -r requirements.txt` (open3d, opencv-python, numpy, python-dotenv). Copy `.env.example` to `.env`. No tests, linter or build configured yet (`src/test/` is empty).

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
- Planned: vision model tags photos by order; floor plan standard schema; `.esx` export.

## Code Style (from `docs/code-instructions.md`)

- Always OOP.
- Beginner-friendly: simple, readable code; beginner-friendly class/function/variable names.
- Avoid heavy comments.
- Use spaces/whitespace generously.
