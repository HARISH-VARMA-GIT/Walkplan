# Walkplan 🤳

Walkplan turns room captures into floor plans. Every output plan has walls, doors, windows and ceiling height, and for videos also the spoken damage marked on the walls. Each plan is saved as `floor_plan.json` plus a `floor_plan.svg` you can open in a browser.

| Input | What you give it | What you get |
|---|---|---|
| **Photos** | One folder per room, about 8 photos taken from the middle of the room while turning clockwise | Room outline, doors, windows, ceiling height |
| **Video** | One walkthrough video, where you talk about any damage. It can cover one room or several | The same, plus every damage you mention, placed on the plan with photos. Several rooms come out as one combined plan |
| **LiDAR** | One [Stray Scanner](https://docs.strayrobots.io/apps/scanner/) capture from an iPhone/iPad Pro, one room or a whole flat | Rooms split automatically, wall lengths to about ±1 cm, doors, windows, a ceiling height per room, and which rooms connect |

Accuracy results against tape measurements are in the reports (section 3).

**Contents**
1. [Installation and code setup](#1-installation-and-code-setup)
2. [Capturing and running](#2-capturing-and-running)
3. [Reports and analysis](#3-reports-and-analysis)
4. [Outputs and measurements](#4-outputs-and-measurements)

---

## 1. Installation and code setup

### 1.1 What you need

- **An NVIDIA GPU with at least 8 GB of memory**, and a recent driver: version 580 or newer, because the code uses CUDA 13 PyTorch. Check with `nvidia-smi`.
- **An OpenAI API key.** It is used for speech-to-text and finding damage in videos, and for the optional `--layout llm` mode for photos.
- **About 25 GB of free disk space:** the Docker image is roughly 15 GB, and the AI models download another ~7 GB on first run.
- Then **either Docker (1.2, recommended)** or **Python 3.11 (1.3)**.

### 1.2 Setup with Docker (recommended)

**Install Docker with GPU support:**
- **Windows:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL 2 backend. GPU support is built in when your NVIDIA driver is up to date.
- **Linux:** install Docker Engine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Check that Docker can see the GPU:

```
docker run --rm --gpus all ubuntu nvidia-smi
```

**Get the code and add your API key:**

```
git clone <repo-url> walkplan
cd walkplan
cp .env.example .env        # Windows PowerShell: copy .env.example .env
```

Open `.env` and set `OPENAI_API_KEY=sk-...`. Leave the `HF_HOME` / `TORCH_HOME` lines commented out when using Docker.

**Build the image** (the first build takes 10–30 minutes):

```
docker compose build
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

### 1.3 Setup without Docker (local Python)

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

- Run every command **from the repo root**, because `requirements.txt` installs MoGe from `./src/MoGe`.
- Do not install `opencv-python` next to `opencv-python-headless`; they clash.
- Models are cached in `<repo>/.cache/`.

### 1.4 Troubleshooting installation

| Problem | Fix |
|---|---|
| `could not select device driver "nvidia"` / no GPU in Docker | Install the NVIDIA Container Toolkit (Linux) or update Docker Desktop and the NVIDIA driver (Windows). Check with `docker run --rm --gpus all ubuntu nvidia-smi` |
| `CUDA driver version is insufficient` | Update the NVIDIA driver to 580 or newer |
| Model download fails with a memory error | Add `HF_HUB_DISABLE_XET=1` to `.env` and run again; downloads resume |
| `ffmpeg not found` (local setup) | Install ffmpeg, or set `FFMPEG_PATH` in `.env` |
| `cv2` errors (local setup) | `pip uninstall opencv-python opencv-python-headless`, then `pip install opencv-python-headless==4.10.0.84` |

---

## 2. Capturing and running

> **Follow the step-by-step capture manual: [docs/capture-and-run-manual.md](docs/capture-and-run-manual.md).** It covers which app to install, phone settings, how to take the photos (order and turning), how to record the video and LiDAR scan, how to move the files to the computer, and the exact command to run. Below is a short summary.

### 2.1 Where to put the captures

The `src/inputs/` folder is not in git; create it yourself:

```
src/inputs/
├── images/
│   └── room1/          ← photos of ONE room: 01.jpg … 08.jpg
├── videos/
│   └── house_walk/     ← ONE video file per folder (mp4 / mov)
│       └── walk.mp4
└── lidar/
    └── flat_scan/      ← ONE Stray Scanner export, unzipped (the folder may sit one level deeper)
        ├── rgb.mp4
        ├── depth/  confidence/
        └── camera_matrix.csv  odometry.csv
```

### 2.2 Capture in short

- **Photos:**
  1. Stand in the middle of the room. Photo 1 faces the entrance door.
  2. Turn **clockwise** about 45° per photo, 8 photos in total.
  3. Use landscape and the 0.5× lens, and send the original files.
- **Video:**
  1. Use landscape and the 0.5× lens, and walk slowly about 1 m from the walls.
  2. For each damage, stop with it in the centre of the frame and say what it is.
  3. For several rooms, say each room's name as you enter it.
- **LiDAR:**
  1. Install Stray Scanner and record all rooms in **one** continuous recording.
  2. Walk into every room and tilt up at each ceiling once.

### 2.3 Run (one command per capture)

**Docker:**

```
docker compose run -it --rm walkplan --images src/inputs/images/room1 --room-type bedroom
docker compose run -it --rm walkplan --video src/inputs/videos/house_walk
docker compose run -it --rm walkplan --lidar src/inputs/lidar/flat_scan
```

**Local Python:**

```
# Windows
.venv\Scripts\python.exe src\app.py --images src\inputs\images\room1 --room-type bedroom
.venv\Scripts\python.exe src\app.py --video src\inputs\videos\house_walk
.venv\Scripts\python.exe src\app.py --lidar src\inputs\lidar\flat_scan

# Linux
.venv/bin/python src/app.py --video src/inputs/videos/house_walk
.venv/bin/python src/app.py --lidar src/inputs/lidar/flat_scan
```

The first run also downloads the models (~7 GB), so later runs start quickly. Every step saves its result, so a second run on the same input only redoes what changed.

| Option | Meaning |
|---|---|
| `--images <folder>` | Folder of photos of one room |
| `--video <folder or file>` | A walkthrough video, or a folder holding one |
| `--lidar <folder>` | A Stray Scanner LiDAR capture (one or several rooms) |
| `--room-type bedroom` | Room label: `bedroom`, `kitchen`, `bathroom`, `living_room`, … For multi-room videos it is detected from speech; multi-room LiDAR rooms are `other` |
| `--output <folder>` | Where results go. Default: `output/<name>` for photos, `output/<name>_video` for videos, `output/<name>_lidar` for LiDAR |
| `--max-frames 24` | Video only: frames used per room. More frames give better coverage but need more GPU memory; 25 fits on 8 GB |
| `--redo-video` | Video only: transcribe, find damage and pick frames again |
| `--redo-photos` | Measure the photos or frames again. Depth and camera poses stay cached |
| `--redo-layout` | Rebuild the room outline |
| `--redo-lidar` | LiDAR only: fuse the depth frames and search for doors and windows again |
| `--no-openings` | LiDAR only: skip the camera search for doors and windows (fast, no GPU); open doorways are still found from the walls |
| `--layout geometry\|llm` | Photos only. `geometry` (default) uses camera poses; `llm` lets a vision model guess the layout |

### 2.4 What a run produces

In `output/<name>/` (photos), `output/<name>_video/` (video) or `output/<name>_lidar/` (LiDAR):

| File | What it is |
|---|---|
| `floor_plan.svg` | **The plan.** Open it in a browser. It shows:<ul><li>walls with lengths ± uncertainty;</li><li>door and window tags, with a "Doors and windows" table underneath;</li><li>room area and ceiling height;</li><li>damage markers and a damage table (video)</li></ul> |
| `floor_plan.json` | The same plan as data. Every number has `value`, `low`, `high` and `method` (`measured` / `estimated` / `assumed`) |
| `overlays/` | Each photo or frame with the detected walls, corners, doors and windows drawn on it. Look here first when a plan looks wrong |
| `damage/` | Video: stills of each damage with a box around it |
| `transcript.json`, `damage_mentions.json`, `rooms.json` | Video: what was said, the damage list and the room list taken from the speech |
| `rooms/NN_name/` | Multi-room video: the full results for each room. The combined plan is in the top folder |
| `layout.json` | Room outline. You can edit it by hand and re-run to rebuild the plan |
| `plan_debug.png`, `points.ply` | LiDAR: top view of the point cloud with rooms and openings drawn on it, and the fused point cloud (open in MeshLab or CloudCompare) |
| `openings/` | LiDAR: video frames with the doors and windows that were found |

### 2.5 Troubleshooting runs

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Close other GPU programs. Lower `--max-frames` (e.g. 18) |
| `Not enough walls found in two directions` | The photos or video do not show enough walls. Re-capture following the manual |
| `No Stray Scanner capture … found` | The LiDAR capture is still zipped or nested too deep. Unzip it so `odometry.csv` is inside `src/inputs/lidar/<name>/<code>/` |
| No damage found in a video | Check `transcript.json`: the damage must be said out loud. Check `damage/` to see what was picked |

---

## 3. Reports and analysis

For the full write-up and the accuracy analysis, see:

- **[docs/Walkplan Project Report.docx](docs/Walkplan%20Project%20Report.docx)**: project report covering the approach, architecture, how each tier was built, fixes, known failures and takeaways.
- **[docs/Walkplan Accuracy Report.pptx](docs/Walkplan%20Accuracy%20Report.pptx)**: accuracy report with the measurements against ground truth.

Technical docs in the repo:
- [docs/project-report.md](docs/project-report.md): the project report in Markdown
- [docs/photo-pipeline.md](docs/photo-pipeline.md): how the photo tier works, step by step
- [docs/video-pipeline.md](docs/video-pipeline.md): video tier, damage and multi-room
- [docs/lidar-pipeline.md](docs/lidar-pipeline.md): LiDAR tier, room splitting, doors and windows
- [docs/spec.md](docs/spec.md): project goals and requirements
- [CLAUDE.md](CLAUDE.md): short code map for developers

---

## 4. Outputs and measurements

All the outputs (floor plans, debug images, point clouds) and the tape/laser measurements used for the accuracy report are in this Google Drive folder:

**https://drive.google.com/drive/folders/1sLxFSPTfwtpSULcIx9zARBCTPJpoI1t9?usp=drive_link**

---

## Models and licences

These models download automatically on first run:
- [MapAnything](https://github.com/facebookresearch/map-anything) camera poses: `facebook/map-anything-apache`, Apache 2.0
- [MoGe-2](https://github.com/microsoft/MoGe) depth: MIT
- [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) and [SAM](https://huggingface.co/facebook/sam-vit-base) for doors, windows and damage

Project licence: MIT (see [LICENSE](LICENSE)).
