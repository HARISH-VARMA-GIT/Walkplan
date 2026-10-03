# Video tier: room walkthrough video → floor plan + damage

This document explains how Walkplan turns one walkthrough video of a room into a floor plan (`FloorPlan` JSON + SVG) with damage marked on it. The floor plan part reuses the photo pipeline (`docs/photo-pipeline.md`), and frames taken from the video are treated as photos.

> **Status (2026-10-03, branch `features/video-processing`):** runs end to end on `src/inputs/videos/room1` (87 s, 25 frames). The room comes out 4.08 × 4.96 m, 18.7 m², ceiling 3.13 m; the photo tier gives 18.2 m² for the same room. Both spoken damages (water stain by the switchboard, spots on the wall) are found and placed on the wall near the window corner. **Not yet checked against tape or real damage positions.**

---

## 1. The idea

1. **Speech:** the speaker says what damage they see. Whisper turns the audio into text with timestamps, and an LLM lists each damage with the time it was mentioned.
2. **Frames:** sharp frames spread over the whole video are used to build the plan. Extra frames from the moments damage is mentioned show the damage.
3. **Floor plan:** all frames go through the photo pipeline: MapAnything poses, MoGe depth, walls, layout and solver.
4. **One lens:** all frames come from one camera, so they share one field of view. That is the median of MapAnything's per-frame estimates, and it is given to MoGe. Without it MoGe guesses a different FOV per frame (49–90° on room1, true ≈105°) and every distance is off.
5. **Damage position:** in each damage frame, Grounding DINO proposes candidate boxes and a vision LLM picks the one that matches what the speaker said. SAM outlines it, and its pixels are projected onto the wall plane, or onto the floor or ceiling, in metres. The solver's camera pose then places it on the plan as a wall, a distance along that wall and a height.

```
video
  ├─ 1 audio        ffmpeg → audio.m4a (mono 16 kHz)
  ├─ 2 transcript   OpenAI whisper-1 (verbose_json) → transcript.json, segments + words with times
  ├─ 3 damage text  LLM → damage_mentions.json (X1, X2 … class, time, surface, description, search words)
  ├─ 4 frames       ffmpeg at 2 fps → sharpest frame per time slot (layout)
  │                 + 2 sharpest frames per damage mention (damage) → frames/, frames.json
  ├─ 5 photo tier   poses (+ shared FOV) → depth with that FOV → walls, openings
  │                 + damage: DINO candidates → vision LLM picks box → SAM → wall/floor
  ├─ 6 assemble     walls, openings, damage (wall + offset + height, or floor/ceiling point)
  └─ 7 render       floor_plan.svg with damage markers + legend, damage/*.jpg stills
```

---

## 2. Setup

Same venv as the photo tier (`.venv\Scripts\python.exe -m pip install -r requirements.txt`). Extra needs:

- **ffmpeg.** Found from `FFMPEG_PATH`, then on PATH (`winget install ffmpeg`), then from the `imageio-ffmpeg` pip package.
- **`OPENAI_API_KEY`** for speech-to-text and the damage step.

`.env` keys:

| Key | Default | Used for |
|---|---|---|
| `TRANSCRIBE_MODEL_NAME` | `whisper-1` | Speech-to-text. Must give timestamps (`gpt-4o-transcribe` does not) |
| `DAMAGE_MODEL_NAME` | unset (uses chain `LLM_DAMAGE_MODEL_CHAIN`, default `gpt-4.1,gpt-4o`) | Finding damage in the transcript |
| `DAMAGE_VISION_MODEL_NAME` | unset (chain `LLM_DAMAGE_VISION_MODEL_CHAIN`, default `gpt-4.1,gpt-4o`) | Picking the damage box in a frame |
| `FFMPEG_PATH` | unset | ffmpeg binary if it is not on PATH |

---

## 3. Recording the video

1. Use landscape, the 0.5× ultra-wide lens, and walk slowly. One room per video, about 1–2 minutes.
2. Start facing the entrance door, then walk **clockwise** around the room about 1 m from the walls. Point the camera at the walls with some floor visible. Every wall, door and window should be on screen at some point.
3. **For each damage, stop.** Hold the camera 1–2 m away with the damage in the **centre** of the frame for about 3 seconds, and say what it is and where it is *while it is on screen*: "Water stain on this wall, under the window, about 30 cm wide."
4. Speak clearly; any language Whisper understands works.
5. Put the file in its own folder per room: `src/inputs/videos/<room>/<file>.mp4`.

---

## 4. Running it

```
.venv\Scripts\python.exe src\app.py --video src\inputs\videos\room1 --room-type bedroom
```

| Option | Meaning |
|---|---|
| `--video <folder or file>` | A folder uses the first video in it. The room name is the folder name, or the file name without extension |
| `--output <folder>` | Default `output/<room>_video` (kept apart from the photo output) |
| `--max-frames` | Total frames sent to the photo pipeline, default 24, damage frames included. More frames give better coverage but use more GPU memory |
| `--redo-video` | Transcribe, find damage and pick frames again. This also re-measures the photos |
| `--redo-photos`, `--redo-layout` | Same as the photo tier |
| `--room-type` | Label in the plan |

Caching: `transcript.json`, `damage_mentions.json` and `frames.json` are reused unless you pass `--redo-video`. You can edit `damage_mentions.json`, for example to fix a class or add `search_terms`, then run with `--redo-photos` so the damage frames are measured again. Changing times needs `--redo-video`, but that re-runs the transcript and overwrites your edits, so in practice keep the times.

---

## 5. Where the code lives

| Step | File / class | Output |
|---|---|---|
| Orchestration | `src/modules/video_processing/main.py` `VideoProcessingService` | – |
| ffmpeg | `src/utils/ffmpeg_runner.py` `FfmpegRunner` (duration, has audio, run) | – |
| Audio | `video_processing/audio_extractor.py` `AudioExtractor` | `audio.m4a` |
| Speech-to-text | `video_processing/transcriber.py` `Transcriber` | `transcript.json` |
| Damage from speech | `video_processing/damage_finder.py` `DamageFinder`, prompt `DAMAGE_PROMPT` in `src/utils/prompts.py`, schema `src/models/video_models.py` | `damage_mentions.json` |
| Frames | `video_processing/frame_selector.py` `FrameSampler` (2 fps candidates with real `pts_time`, sharpness = Laplacian variance), `FrameSelector` | `frame_candidates/`, `frames/`, `frames.json` |
| Floor plan | `image_processing/main.py` `build_floor_plan_from_paths(..., capture_tier="video", damage_requests, damage_details)` | everything from the photo tier |
| Shared FOV | `image_processing/main.py` `shared_fov_from_poses` (MapAnything intrinsics), `DepthEstimator.estimate(..., fov_x_deg)` | log line "Same camera for all frames" |
| Damage in a frame | `image_processing/damage_measurer.py` `DamageMeasurer`, `damage_box_picker.py` `DamageBoxPicker` (prompt `DAMAGE_BOX_PROMPT`) | `photo_geometry.json` → `damages` per photo, `damage/X1_P07.jpg` |
| Damage on the plan | `image_processing/room_assembler.py` `build_damages` | `floor_plan.json` → `rooms[0].damage` |
| Drawing | `floor_plan_generator/svg_creator.py` `draw_damage`, `draw_damage_legend` | `floor_plan.svg` |

Ported from humantic `deep_process`, rewritten as classes: ffmpeg helpers, audio extraction, `pts_time` frame timestamps from `showinfo`, and the timed-transcript prompt format with the BEGIN/END DATA guard.

---

## 6. How damage is placed

1. **Find it in the frame.** Grounding DINO searches with a low threshold for the mention's `search_terms` plus generic words (stain, crack, spot, mark, hole, peeling paint). It keeps up to 8 boxes under 40% of the image.
   - A vision LLM sees the frame with numbered boxes plus the speaker's words, and picks the box that shows the damage. This step is needed: DINO alone often boxes curtains or mirror reflections. SAM then outlines the chosen box. Position is `measured`.
   - If no box fits, the LLM gives the 3×3 grid cell of the damage, and the position is `estimated`. If it says "not visible", the frame is skipped.
   - With no OpenAI key, the top DINO box is used.
2. **Which surface.** A plane is fitted to the MoGe 3D points of the damage.
   - Normal pointing up or down: floor or ceiling.
   - Otherwise: the photo wall whose plane is within 25 cm.
   - If neither applies, the speaker's surface hint is used.
3. **Measure.** Pixels are cast as rays onto the wall plane (the same method as door sizes). That gives position along the wall, width, centre height above the floor, and area (bounding size × mask fill).
4. **Onto the plan.**
   - The frame's camera pose (from the solver, or the MapAnything placement) moves the point into the room frame.
   - Wall: the photo-wall match from the layout, or the nearest plan wall, gives `wall_id`, `offset_m` (from the wall start) and `point`.
   - Floor/ceiling: `point` = x,y in the plan.
   - With two damage frames, the median is used.

`Damage` fields in `floor_plan.json`:
- From the transcript: `damage_class`, `severity`, `description`, `quote`, `notes` (the spoken location), `video_time_s`.
- From geometry: `surface_type`, `wall_id`, `offset_m`, `height_m`, `width_m`, `point`, `extent_m2`.
- Also: `source_frames` (stills), `location_method` (`measured` = detected; `assumed` = camera centre), `confidence`.

In the SVG, each damage is a red circle with its id, placed inside the room next to its wall. A dashed outline means the position is not `measured`. The legend under the plan lists every damage.

---

## 7. Known limitations

- **Damage detection:** DINO has to propose a box on the damage for a `measured` position. Faint marks may only get a grid cell (`estimated`). The `damage/*.jpg` stills show what was picked, and the log lists the LLM's reason.
- **Camera height swings when the phone tilts** (1.0–2.3 m on room1). The MapAnything pose height follows the phone's pitch, so damage and opening heights from tilted frames can be off by a few tens of cm.
- **Damage mentioned while the camera points elsewhere** gets the wrong frame. Point first, then talk.
- **MapAnything runs all frames together:** GPU memory grows with `--max-frames`. 25 frames fit on 8 GB.
- **Windows memory:** the 4.9 GB MapAnything checkpoint is built on the meta device and copied to the GPU tensor by tensor (`PoseEstimator.build_model_low_memory`). The standard `from_pretrained` needs about 10 GB of free commit memory and crashed (exit 139) on this machine.
- **Video frames are blurrier than photos.** Walk slowly. The sharpest frame in each slot is used.
- Speech-to-text is OpenAI only. A video with no speech still gives a floor plan, just with no damage.
- Everything from the photo tier also applies: Manhattan rooms only, one room, door detection noise.
