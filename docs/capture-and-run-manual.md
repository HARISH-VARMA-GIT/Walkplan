# Walkplan – Capture and Run Manual

This is the manual to follow when you capture a space and run Walkplan on it. Please follow it step by step, exactly as written.

There are three ways to capture. Do only the one you want to test:

| Tier | App on the phone | Phone needed | Time on site |
|---|---|---|---|
| A. Photos | Built-in **Camera** app (Photo mode) | Any iPhone 15 or newer | About 1 minute per room |
| B. Video | Built-in **Camera** app (Video mode) | Any iPhone 15 or newer | About 1–2 minutes per room |
| C. LiDAR | **Stray Scanner** (free, App Store) | iPhone 12 Pro or newer **Pro** model, or iPad Pro (2020 or newer) | About 1–2 minutes per room, one recording for the whole space |

Part 1 is the one-time installation on the computer. Parts 2–4 are the three capture tiers. Part 5 is how to run the pipeline and Part 6 is what you get.

---

## Part 1 – One-time installation on the computer

### 1.1 What the computer needs

- Windows 10/11 or Linux, with an **NVIDIA GPU with at least 8 GB memory**. Check with `nvidia-smi`; the driver version must be 580 or newer.
- About **25 GB free disk space**.
- Internet for the first run: the AI models (about 7 GB) download automatically once.
- An **OpenAI API key**. It is used for video speech-to-text and damage. The photo and LiDAR tiers run without it.

### 1.2 Get the code and add the key

```
git clone <repo-url> walkplan
cd walkplan
copy .env.example .env          (Linux: cp .env.example .env)
```

Open `.env` in Notepad and put your key after `OPENAI_API_KEY=`, then save.

### 1.3 Install – choose ONE option

**Option 1: Docker (recommended).**
1. Install Docker Desktop (Windows, with the WSL 2 backend) or Docker Engine plus the NVIDIA Container Toolkit (Linux).
2. Check that Docker can see the GPU:

   ```
   docker run --rm --gpus all ubuntu nvidia-smi
   ```

3. Build the image once (10–30 minutes):

   ```
   docker compose build
   ```

**Option 2: Local Python (Windows).**
1. Install Python 3.11 and ffmpeg (`winget install ffmpeg`).
2. From the `walkplan` folder, run:

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

Run every command from the `walkplan` folder.

### 1.4 Make the input folders

Create this structure inside the `walkplan` folder. It is not in git:

```
src/inputs/
├── images/
├── videos/
└── lidar/
```

---

## Part 2 – Tier A: Photos

### 2.1 Phone settings (once)

Open the Camera app and set:
1. **0.5×** lens: tap "0.5" above the shutter button.Wide mode
2. Hold the phone in **landscape** (sideways) for every photo.

### 2.2 How to take the photos (8 photos per room)

Do this for **each room separately**:

1. Switch on all the room lights and open the curtains.
2. Stand **near the middle of the room**, on open floor.
3. **Photo 1: face the door you entered from.**
4. Hold the phone at **chest height**, in landscape, at 0.5×. Tilt it **slightly up**, so that the photo shows **both** the line where the walls meet the floor (at the bottom) **and** the line where the walls meet the ceiling (at the top). The ceiling line is how the ceiling height is measured.
5. Take the photo.
6. **Turn clockwise (to your right) by about 45°** – one-eighth of a full turn. Turn on the spot; your feet stay where they are.
7. Take the next photo. Repeat until you have **8 photos**, which brings you back to the door.


### 2.3 Move the photos to the computer

1. Send the **original files**:
   - **AirDrop** to a Mac;
   - **USB cable** to Windows (the photos are in DCIM);
   - or **iCloud Photos → Download original**.

   **Do not use WhatsApp or email**: they shrink the photos.
2. Make **one folder per room** and put that room's 8 photos in it:

   ```
   src/inputs/images/kitchen/IMG_1001.HEIC … IMG_1008.HEIC
   src/inputs/images/bedroom/IMG_1009.HEIC … IMG_1016.HEIC
   ```

3. Keep the camera's file names. They sort in the order you took the photos, and **the order matters** (photo 1 = facing the door, then clockwise).

### 2.4 Run (one command per room folder)

```
Docker:  docker compose run --rm walkplan --images src/inputs/images/kitchen --room-type kitchen
Local:   .venv\Scripts\python.exe src\app.py --images src\inputs\images\kitchen --room-type kitchen
```

`--room-type` is one of: `bedroom`, `bathroom`, `kitchen`, `living_room`, `dining_room`, `hallway`, `closet`, `laundry`, `office`, `garage`, `stairs`, `entry`, `other`.

Result: `output/kitchen/floor_plan.svg`. At the photo tier each room folder currently gives its own room plan; the folders are not stitched into one property plan yet.

---

## Part 3 – Tier B: Video

### 3.1 Phone settings (once)

1. Camera app → **Video** mode, not Cinematic, not Slo-mo.
2. Settings → Camera → Record Video → **1080p at 30 fps**.
3. **0.5×** lens, **landscape**. **Action mode off.**

### 3.2 How to record

1. Lights on, curtains open. Start the recording **standing at the entrance door, facing into the room**.
2. Walk **slowly**, about half your normal speed, **clockwise** around the room, about **1 m away from the walls**. Point the camera at the walls with some floor visible at the bottom of the screen.
3. Every wall, door and window must be on screen at some moment.
4. **Ceiling:** once in every room, stop, slowly tilt the phone up until the ceiling and the top of the wall fill the screen, hold for 2 seconds, then tilt back down.
5. **Damage:** for every damage:
   - **stop** 1–2 m away;
   - put the damage in the **centre of the screen** and hold still for about **3 seconds**;
   - **while it is on screen**, say out loud what it is and where it is, for example: "Water stain on this wall, under the window, about 30 centimetres wide."
6. **Several rooms in one video** (allowed):
   - finish the first room completely;
   - say the next room's name **out loud** as you enter it ("now the kitchen");
   - walk **slowly through the doorway** and film the door frame from both sides;
   - then do the new room as in steps 2–5.
7. Stop the recording. One room takes about 1–2 minutes.


### 3.3 Move the video to the computer

AirDrop or USB cable, **original file** (not WhatsApp). Put **one video per folder**:

```
src/inputs/videos/flat_walk/IMG_2001.MOV
```

### 3.4 Run (one command per video)

```
Docker:  docker compose run --rm walkplan --video src/inputs/videos/flat_walk
Local:   .venv\Scripts\python.exe src\app.py --video src\inputs\videos\flat_walk
```

Result: `output/flat_walk_video/floor_plan.svg`. With several rooms, the combined plan is at the top of that folder and each room's details are in `rooms/`.

---

## Part 4 – Tier C: LiDAR (Stray Scanner)

### 4.1 Install the app (once)

1. On an iPhone **Pro** or iPad Pro, open the **App Store** and search **"Stray Scanner"** (by Stray Robots). It is free. Install it.
2. Open it and allow **camera** access. Keep the default settings.

### 4.2 How to record – the whole space in ONE recording

1. Switch on all lights. Open the doors between the rooms you want to scan.
2. Start in the first room. Tap the **record** button.
3. Hold the phone in **portrait** (upright), at chest height, pointed at the walls, **1–2 m away from them**.
4. In each room:
   - Walk **slowly** around the room. Sweep each wall from **floor to head height** once.
   - **Walk inside every room.** A room the camera never enters is left out of the plan.
   - **Ceiling:** once per room, tilt the phone up to the ceiling for 2 seconds and back down. Without this, the ceiling height of that room is reported as "not seen".
   - Point at every door and window for a moment.
5. **Moving between rooms:** walk **slowly through the doorway** and point at the door frame from both sides. **Do not stop the recording** between rooms. Everything must be one continuous recording; two separate recordings are not joined.
6. When all rooms are done, tap **stop**. A flat of 4–5 rooms takes about 3–4 minutes.


**Mirrors and glass:** point at them only briefly and at an angle. LiDAR sees through glass and "behind" mirrors.

### 4.3 Move the capture to the computer

1. On the iPhone, open the **Files** app → **On My iPhone** → **Stray Scanner**. Each recording is a folder with a short code name (for example `d0ecafeb94`).
2. Long-press that folder → **Compress**. This makes `d0ecafeb94.zip`.
3. Send the zip to the computer: AirDrop, USB cable, or upload to Google Drive / OneDrive and download.
4. On the computer, **unzip** it (Windows: right-click → Extract All) into its own folder:

   ```
   src/inputs/lidar/flat/d0ecafeb94/
                              ├── rgb.mp4
                              ├── depth/
                              ├── confidence/
                              ├── camera_matrix.csv
                              └── odometry.csv
   ```

   The folder **must be unzipped**. Put **only one capture** inside `src/inputs/lidar/flat/`.

### 4.4 Run (one command per capture)

```
Docker:  docker compose run --rm walkplan --lidar src/inputs/lidar/flat
Local:   .venv\Scripts\python.exe src\app.py --lidar src\inputs\lidar\flat
```

Rooms are found automatically, so one capture can hold one room or the whole flat. Add `--room-type bedroom` only when the scan has a single room. Add `--no-openings` for a fast run without the door/window camera search; open doorways are still found.

Result: `output/flat_lidar/floor_plan.svg`.

---

## Part 5 – Running: what to expect

| Tier | First run (models download) | Later runs |
|---|---|---|
| Photos, one room | + about 10 minutes for downloads | a few minutes |
| Video, one room | + downloads | several minutes (speech, frames, then the photo pipeline) |
| LiDAR, whole flat | + downloads | about 1 minute for the plan, plus about 5 minutes for the door/window search |

- Each run prints `Floor plan saved to …` at the end.
- Every step is cached, so running the same command again is fast.
- To force a fresh run, add:
  - `--redo-photos` (photos),
  - `--redo-video` (video), or
  - `--redo-lidar` (LiDAR).

---

## Part 6 – What you get

Everything is in `output/<name>/` (photos), `output/<name>_video/` or `output/<name>_lidar/`.

| File | What it is |
|---|---|
| `floor_plan.svg` | **The plan.** Open it in any web browser. It shows:<ul><li>rooms with area and ceiling height;</li><li>walls with length ± uncertainty;</li><li>door and window tags;</li><li>a "Doors and windows" table underneath;</li><li>damage markers and a damage table (video).</li></ul> |
| `floor_plan.json` | Same plan as data. Every number has `value`, `low`, `high` and `method` (`measured` / `estimated` / `assumed`) |
| `overlays/` | Photo/video tiers: each photo with the detected walls, doors and windows |
| `damage/` | Video tier: stills of each damage |
| `plan_debug.png` | LiDAR tier: top view of the scan with the rooms and doors drawn on it |
| `openings/` | LiDAR tier: frames where doors and windows were found |

**How to read the plan:**
- **±** after a length is the uncertainty range.
- "**sill**" on a window is the height from the floor to the bottom of the window.
- "**ceiling not seen**" means the ceiling was not captured; the system does not invent a height.

---

## Part 7 – If something goes wrong

| Message / problem | What to do |
|---|---|
| `Not enough walls found` (photos/video) | The photos or video did not show enough walls. Re-capture following Part 2 or 3 |
| `No Stray Scanner capture … found` | The LiDAR folder is still a zip, or it is nested two levels deep. Unzip so that `odometry.csv` sits in `src/inputs/lidar/<name>/<code>/` |
| `Found 2 captures` | Put only one capture folder per `src/inputs/lidar/<name>/` |
| `CUDA out of memory` | Close other programs using the GPU and run again. For video add `--max-frames 18` |
| No damage in the video plan | The damage was not said out loud while on screen. Check `transcript.json` |
| A room is missing in the LiDAR plan | The camera never walked into that room. Re-record and walk inside it |
| Model download fails with a memory error | Add `HF_HUB_DISABLE_XET=1` to `.env` and run again; downloads resume |
