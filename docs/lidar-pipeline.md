# LiDAR pipeline

The LiDAR tier turns one [Stray Scanner](https://docs.strayrobots.io/apps/scanner/) capture from an iPhone/iPad Pro into a floor plan. The capture can cover one room or a whole flat. Each room comes out with:

- walls, with lengths and ± uncertainty;
- doors and windows;
- its own ceiling height;
- links to the neighbouring rooms.

The output uses the same `floor_plan.json` / `floor_plan.svg` format as the photo and video tiers.

Accuracy has not been checked against tape measurements yet.

## 1. Run

```
.venv\Scripts\python.exe src\app.py --lidar src\inputs\lidar\<name> [--room-type bedroom] [--redo-lidar] [--no-openings]
```

| Option | Meaning |
|---|---|
| `--lidar <folder>` | A Stray Scanner capture folder, or a folder holding exactly one |
| `--room-type` | Used only when the scan holds a single room; with several rooms every room is `other` |
| `--redo-lidar` | Fuse the depth frames and search for doors/windows again (both are cached) |
| `--no-openings` | Skip the camera search for doors and windows. It is fast and needs no GPU; only open doorways found in the walls are reported |

Results go to `output/<name>_lidar/` unless you pass `--output`.

A capture folder holds:

- `rgb.mp4`;
- `depth/000000.png …` (16-bit millimetres, 256×192);
- `confidence/` (0–2);
- `camera_matrix.csv` (intrinsics for the RGB size);
- `odometry.csv`, with one camera pose per frame: `timestamp, frame, x, y, z, qx, qy, qz, qw`.

## 2. Steps

| Step | File | What happens |
|---|---|---|
| Read | `capture_reader.py` | Poses, intrinsics, depth and confidence. Stray poses are camera-to-world **in OpenCV camera axes** (x right, y down, z forward), so no axis flip is applied. The old scripts flipped them, which smeared the cloud |
| Fuse | `point_cloud_builder.py` | Fuses up to 900 depth frames (confidence 2, 0.2–4.5 m) into 2 cm voxels. Normals come from depth-image gradients and face the camera |
| Plan frame | `plan_frame.py` | The floor is the biggest up-facing height peak below the camera. The ceiling is the biggest down-facing peak 2–4.5 m above the floor, and it must hold at least 5% as many points as the floor (otherwise it is not reported). The wall angle is the main angle of wall normals, taken mod 90°. Plan coordinates are (x, −z) rotated so walls are axis-aligned. Using (x, −z) keeps the plan from being mirrored |
| Grids | `plan_grid.py` | Builds 5 cm rasters: free space (floor plus flat surfaces below the ceiling), and the lowest and highest wall point per cell for x-facing and y-facing walls (0.15–2.0 m above the floor) |
| Wall lines | `room_segmenter.py` `WallLineFinder` | 1-D peaks of wall-point positions, found separately for each facing direction. A line counts only if at least 0.4 m of its length has wall points spanning ≥0.8 m in height. This drops sofas, beds and counters |
| Rooms | `room_segmenter.py` `RoomSegmenter` | **Barrier:** tall-wall cells, plus wall-line runs, plus door-sized gaps (≤1.1 m) closed **only along the wall direction**. Closing only along the wall stops parallel corridor walls from being bridged.<br>**Rooms:** connected free space between barriers becomes rooms. Pieces under 0.8 m² are dropped, and so are areas the camera never entered (balconies seen through glass). Closets under 3 m² only need the camera within 1.5 m.<br>**Outline:** notches (unseen floor under furniture) are filled. The pixel outline is traced, each edge snapped to the nearest wall line within 0.25 m, and small jogs removed (`GeometryLayoutBuilder`) |
| Doors and windows (camera) | `opening_detector.py` | Picks one low-motion, roughly level keyframe per 0.6 s (at most 300). Each is turned upright using gravity, because Stray videos are stored sideways, and shrunk to 1024 px. Grounding DINO and SAM then run on it (same models as the photo tier). Results are cached in `opening_detections.json`. Up to 40 annotated frames go in `openings/` |
| Place openings | `opening_placer.py` | The LiDAR depth at the mask picks the wall; if there is no depth, the first wall the centre ray crosses is used. Mask pixel rays are cast onto that wall plane, giving the offset, width and heights. Size checks reuse `VALID_SIZES`, a door must reach the floor, and a window sill must be 0.2–1.6 m. Openings are grouped across frames: each needs 2 views, or a score of 0.5 or more. Windows are kept only on outside walls (no room behind them) and never on top of a door |
| Doors (geometry) | `opening_placer.py` `gap_openings` | A wall gap of 0.55–1.05 m with floor visible on both sides and no low wall (which would mean a window) becomes a door, with its width measured from the gap. When it matches a camera door, the camera door takes the gap's width |
| Plan | `plan_builder.py` | Builds a `FloorPlan`. Wall length uncertainty comes from the spread of the two end lines (±2σ); edges not snapped to a line get ±0.10 m. Wall thickness is the gap to the facing wall of the next room (default 0.12 m). Each room's ceiling comes from its own ceiling points, falling back to the overall ceiling as `estimated`. Doors get `leads_to` and an adjacency entry |
| Render | `floor_plan_generator`, `debug_drawer.py` | `floor_plan.svg`, plus `plan_debug.png`, a top view of the 1.3–1.9 m wall slice with the floor, camera path, rooms and openings |

## 3. Outputs

| File | What it is |
|---|---|
| `floor_plan.svg` / `floor_plan.json` | The plan |
| `plan_debug.png` | Point-cloud top view with rooms and openings. **Look here first when the plan looks wrong** |
| `points.ply` | Fused point cloud with normals, for viewing in MeshLab or CloudCompare |
| `point_cloud.npz` | Point cloud cache |
| `plan_frame.json` | Floor height, wall angle and overall ceiling |
| `opening_detections.json`, `openings/` | Raw door/window detections and annotated keyframes |
| `openings.json` | Placed openings before they are turned into plan objects |

## 4. Results on the sample captures

| Capture | Rooms | Notes |
|---|---|---|
| `single_scan_with_ceiling` | 5 (27.4, 11.6, 9.4, 6.7, 3.2 m²) | The open living area and corridor form one room, as expected. Ceilings 2.27–3.07 m. Some windows still land on interior walls |
| `single_scan_floor_only` | 7 | Same flat. Ceiling correctly not reported. Walls are less complete, so the corridor merges with a bedroom |
| `single_room` | 2 | The capture mostly shows floor, so few walls; the outline is rough |

Runtime: fusion 15–50 s, door/window search about 5 min for 300 keyframes on an 8 GB GPU, everything else a few seconds.

## 5. Capture tips

- Hold the phone so the walls fill most of the view, about 1–2 m away. Sweep each wall from floor to ceiling height once.
- Tilt up at the ceiling at least once in each room; otherwise the ceiling height is not reported.
- Walk through every doorway and film the door frame from both sides.
- Walk slowly. Fast turns blur the RGB keyframes used to find doors and windows.

## 6. Limits

- Rooms of a multi-room scan are labelled `other` and named `Room 1…`. Room types are not detected yet.
- Open-plan spaces joined by openings wider than 1.1 m become one room.
- A missing wall line (a wall never scanned) leaves that edge at the raster position, ±0.10 m.
- No damage detection on the LiDAR tier yet.
