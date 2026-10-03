# Walkplan 🤳

Walkplan turns room captures into floor plans. Every output plan has walls, doors, windows, ceiling height and, for videos, the spoken damage marked on the walls. Each plan is saved as `floor_plan.json` plus a `floor_plan.svg` you can open in a browser.

| Input | What you give it | What you get |
|---|---|---|
| **Photos** | One folder per room, about 8 photos taken from the middle of the room while turning clockwise | Room outline, doors, windows, ceiling height |
| **Video** | One walkthrough video, where you talk about any damage. It can cover one room or several | The same, plus every damage you mention, placed on the plan with photos. Several rooms come out as one combined plan |

Accuracy has **not** been checked against tape measurements yet.

---

## 1. What you need

- **An NVIDIA GPU with at least 8 GB of memory**, and a recent driver: version 580 or newer, because the code uses CUDA 13 PyTorch. Check with `nvidia-smi`.
- **An OpenAI API key.** It is used for speech-to-text and for finding damage in videos, and for the optional `--layout llm` mode for photos.
- **About 25 GB of free disk space:** the Docker image is roughly 15 GB, and the AI models download another ~7 GB on first run.
- Then **either Docker (section 2, recommended)** or **Python 3.11 (section 3)**.

---

## 2. Setup with Docker (recommended)

### 2.1 Install Docker with GPU support

- **Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL 2 backend. GPU support is built in when your NVIDIA driver is up to date.
- **Linux:** install Docker Engine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Check that Docker can see the GPU:

```
docker run --rm --gpus all ubuntu nvidia-smi
```

### 2.2 Get the code and add your API key

```
git clone <repo-url> walkplan
cd walkplan
cp .env.example .env        # Windows PowerShell: copy .env.example .env
```

Open `.env` and set `OPENAI_API_KEY=sk-...`. Leave the `HF_HOME` / `TORCH_HOME` lines commented out when using Docker.

### 2.3 Build the image

```
docker compose build
```

The first build takes 10–30 minutes, because it downloads PyTorch and the other packages.

### 2.4 Run

Put your captures under `src/inputs/` (see section 4), then:

```
docker compose run --rm walkplan --images src/inputs/images/room1 --room-type bedroom
docker compose run --rm walkplan --video src/inputs/videos/room1
```

Results appear in `output/` on your machine. The first run also downloads the models (~7 GB) into a Docker volume, so later runs start quickly.

Show all options:

```
docker compose run --rm walkplan --help
```

<details>
<summary>Without docker compose</summary>

```
docker build -t walkplan .
docker run --rm --gpus all --env-file .env \
  -v "$(pwd)/src/inputs:/app/src/inputs" \
  -v "$(pwd)/output:/app/output" \
  -v walkplan-cache:/app/.cache \
  walkplan --video src/inputs/videos/room1
```

In Windows PowerShell, use `${PWD}` instead of `$(pwd)` and put the whole command on one line.
</details>

---

## 3. Setup without Docker (local Python)

You need **Python 3.11** and **ffmpeg** on PATH. On Windows, `winget install ffmpeg` installs it; on Ubuntu, `sudo apt install ffmpeg git`.

```
git clone <repo-url> walkplan
cd walkplan

# Windows
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Linux
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env        # then set OPENAI_API_KEY
```

Run every command **from the repo root**, because `requirements.txt` installs MoGe from `./src/MoGe`.

- Do not install `opencv-python` next to `opencv-python-headless`; they clash.
- Models are cached in `<repo>/.cache/`.

Run:

```
# Windows
.venv\Scripts\python.exe src\app.py --images src\inputs\images\room1 --room-type bedroom
.venv\Scripts\python.exe src\app.py --video src\inputs\videos\room1

# Linux
.venv/bin/python src/app.py --video src/inputs/videos/room1
```

---

## 4. Preparing your captures

The `src/inputs/` folder is not in git; create it yourself:

```
src/inputs/
├── images/
│   └── room1/          ← photos of ONE room: 01.jpg … 08.jpg
└── videos/
    └── house_walk/     ← ONE video file per folder (mp4 / mov)
        └── walk.mp4
```

**Photos** (one folder per room):
1. Stand near the middle of the room. Photo 1 faces the entrance door.
2. Turn **clockwise** about 45° per photo, 8 photos in total. Hold the phone level, in landscape, on the 0.5× ultra-wide lens.
3. Send the original files, not WhatsApp copies, which lose quality.

**Video:**
1. Landscape, 0.5× lens. Walk slowly around the room about 1 m from the walls, so every wall, door and window is on screen at some point.
2. **Damage:** stop, hold the damage in the **centre** of the frame for about 3 seconds, and say what it is while it is on screen ("water stain on this wall under the window").
3. **Several rooms in one video:** say the room name as you enter each one ("now the kitchen"). Walk slowly through the doorway and film the door from both sides. Finish one room before moving to the next.

---

## 5. Commands

```
python src/app.py --images <folder> [options]
python src/app.py --video  <folder or file> [options]
```

| Option | Meaning |
|---|---|
| `--images <folder>` | Folder of photos of one room |
| `--video <folder or file>` | A walkthrough video, or a folder holding one |
| `--room-type bedroom` | Room label: `bedroom`, `kitchen`, `bathroom`, `living_room`, … For multi-room videos it is detected from speech |
| `--output <folder>` | Where results go. Default: `output/<name>` for photos, `output/<name>_video` for videos |
| `--max-frames 24` | Video only: frames used per room. More frames give better coverage but need more GPU memory; 25 fits on 8 GB |
| `--redo-video` | Video only: transcribe, find damage and pick frames again |
| `--redo-photos` | Measure the photos or frames again. Depth and camera poses stay cached |
| `--redo-layout` | Rebuild the room outline |
| `--layout geometry\|llm` | Photos only. `geometry` (default) uses camera poses; `llm` lets a vision model guess the layout |

Every step saves its result, so a second run on the same input only redoes what changed.

---

## 6. Outputs

In `output/<name>/` (photos) or `output/<name>_video/` (video):

| File | What it is |
|---|---|
| `floor_plan.svg` | **The plan.** Open it in a browser. Walls with lengths ± uncertainty, doors, windows, room area, ceiling height, red damage markers and a damage legend |
| `floor_plan.json` | The same plan as data. Every number has `value`, `low`, `high` and `method` (`measured` / `estimated` / `assumed`) |
| `overlays/` | Each photo or frame with the detected walls, corners, doors and windows drawn on it. Look here first when a plan looks wrong |
| `damage/` | Video: stills of each damage with a box around it |
| `transcript.json`, `damage_mentions.json`, `rooms.json` | Video: what was said, the damage list and the room list taken from the speech |
| `rooms/NN_name/` | Multi-room video: the full results for each room. The combined plan is in the top folder |
| `layout.json` | Room outline. You can edit it by hand and re-run to rebuild the plan |

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `could not select device driver "nvidia"` / no GPU in Docker | Install the NVIDIA Container Toolkit (Linux) or update Docker Desktop and the NVIDIA driver (Windows). Check with `docker run --rm --gpus all ubuntu nvidia-smi` |
| `CUDA driver version is insufficient` | Update the NVIDIA driver to 580 or newer |
| `CUDA out of memory` | Close other GPU programs. Lower `--max-frames` (e.g. 18) |
| Model download fails with a memory error | Add `HF_HUB_DISABLE_XET=1` to `.env` and run again; downloads resume |
| `ffmpeg not found` (local setup) | Install ffmpeg, or set `FFMPEG_PATH` in `.env` |
| `cv2` errors (local setup) | `pip uninstall opencv-python opencv-python-headless`, then `pip install opencv-python-headless==4.10.0.84` |
| `Not enough walls found in two directions` | The photos or video do not show enough walls. Re-capture following section 4 |
| No damage found in a video | Check `transcript.json`: the damage must be said out loud. Check `damage/` to see what was picked |

---

## 8. More documentation

- [docs/photo-pipeline.md](docs/photo-pipeline.md): how the photo tier works, step by step
- [docs/video-pipeline.md](docs/video-pipeline.md): video tier, damage and multi-room
- [docs/spec.md](docs/spec.md): project goals and requirements
- [CLAUDE.md](CLAUDE.md): short code map for developers

LiDAR scripts (Stray Scanner captures) are in `src/modules/lidar_processing/`; see `CLAUDE.md` for their commands.

## Models and licences

These models download automatically on first run:
- [MapAnything](https://github.com/facebookresearch/map-anything) camera poses: `facebook/map-anything-apache`, Apache 2.0
- [MoGe-2](https://github.com/microsoft/MoGe) depth: MIT
- [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) and [SAM](https://huggingface.co/facebook/sam-vit-base) for doors, windows and damage


Project licence: MIT (see [LICENSE](LICENSE)).
