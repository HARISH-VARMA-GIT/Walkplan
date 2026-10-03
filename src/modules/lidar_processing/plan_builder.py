import numpy as np

from models.floor_plan import Adjacency, CaptureInfo, FloorPlan, Measurement, Opening, Room, Wall
from modules.lidar_processing.opening_placer import DEFAULT_DOOR_HEIGHT_M
from modules.lidar_processing.plan_frame import HORIZONTAL_NORMAL, MAX_CEILING_HEIGHT_M, MIN_CEILING_HEIGHT_M, PlanFrame, PlanFrameFinder
from modules.lidar_processing.plan_grid import PlanGrid
from modules.lidar_processing.point_cloud_builder import PointCloud
from modules.lidar_processing.room_segmenter import RoomFinder


SOURCE = "lidar_planes"
DEFAULT_THICKNESS_M = 0.12
MIN_THICKNESS_M = 0.03
MAX_THICKNESS_M = 0.45
MIN_SHARED_LENGTH_M = 0.3
MIN_CEILING_COVER = 0.2
CEILING_SIGMA_FLOOR_M = 0.01
LEADS_TO_REACH_M = 0.45
TWO_SIGMA = 2.0
MIN_HALF_WIDTH_M = 0.01


class LidarPlanBuilder:

    def __init__(self, cloud: PointCloud, frame: PlanFrame, grid: PlanGrid):
        self.frame = frame
        self.grid = grid
        self.notes = []
        self.frame_finder = PlanFrameFinder()

        top = MAX_CEILING_HEIGHT_M if frame.ceiling_height_m is None else frame.ceiling_height_m + 0.3
        is_ceiling = cloud.normals[:, 1] < -HORIZONTAL_NORMAL
        heights = frame.height(cloud.points[is_ceiling])
        in_range = (heights > MIN_CEILING_HEIGHT_M) & (heights < top)
        self.ceiling_heights = heights[in_range]
        self.ceiling_cells = grid.cell_index(frame.to_plan(cloud.points[is_ceiling][in_range]))

    def measurement(self, value: float, half_width: float, method: str = "measured", source: str = SOURCE) -> Measurement:
        half_width = max(half_width, MIN_HALF_WIDTH_M)
        return Measurement(value=round(value, 3), low=round(value - half_width, 3), high=round(value + half_width, 3),
                           method=method, source=source)

    def room_ceiling(self, room):
        inside = room.mask[self.ceiling_cells[:, 0], self.ceiling_cells[:, 1]]
        heights = self.ceiling_heights[inside]

        if len(heights) == 0:
            return self.fallback_ceiling(room)

        covered = len(set(map(tuple, self.ceiling_cells[inside]))) / max(int(room.mask.sum()), 1)
        peaks = self.frame_finder.height_peaks(heights, MIN_CEILING_HEIGHT_M, MAX_CEILING_HEIGHT_M)
        if covered < MIN_CEILING_COVER or not peaks:
            return self.fallback_ceiling(room)

        best = max(peaks, key=self.frame_finder.count_of)
        half_width = TWO_SIGMA * float(np.hypot(best["spread"], self.frame.floor_spread_m)) / np.sqrt(best["count"])
        return self.measurement(best["height"], half_width + CEILING_SIGMA_FLOOR_M, "measured", "lidar_ceiling")

    def fallback_ceiling(self, room):
        if self.frame.ceiling_height_m is None:
            self.notes.append(f"R{room.index}: ceiling not seen, height not reported")
            return None

        self.notes.append(f"R{room.index}: ceiling not seen, used the height measured in other rooms")
        return self.measurement(self.frame.ceiling_height_m, 0.05, "estimated", "lidar_ceiling_other_rooms")

    def wall_length(self, room, edge) -> Measurement:
        if edge.normal[0] != 0:
            first = room.sigma_of("y", float(edge.start[1]))
            second = room.sigma_of("y", float(edge.end[1]))
        else:
            first = room.sigma_of("x", float(edge.start[0]))
            second = room.sigma_of("x", float(edge.end[0]))

        return self.measurement(edge.length, TWO_SIGMA * float(np.hypot(first, second)))

    def wall_thickness(self, room, edge, rooms: list) -> float:
        best = None

        for other in rooms:
            if other.index == room.index:
                continue

            for other_edge in other.edges:
                if other_edge.normal != (-edge.normal[0], -edge.normal[1]):
                    continue

                gap = float(np.array(edge.normal) @ (edge.start - other_edge.start))
                if not MIN_THICKNESS_M <= gap <= MAX_THICKNESS_M:
                    continue

                low = max(0.0, min(edge.along(other_edge.start), edge.along(other_edge.end)))
                high = min(edge.length, max(edge.along(other_edge.start), edge.along(other_edge.end)))
                if high - low >= MIN_SHARED_LENGTH_M and (best is None or gap < best):
                    best = gap

        if best is None:
            return DEFAULT_THICKNESS_M
        return round(best, 3)

    def build_opening(self, opening_id: str, item: dict) -> Opening:
        width = item["end"] - item["start"]
        width_measurement = self.measurement(width, TWO_SIGMA * item["width_spread"])

        if item["type"] == "door":
            if item["top"] is None:
                height = self.measurement(DEFAULT_DOOR_HEIGHT_M, 0.1, "assumed", "standard_door_height")
            else:
                height = self.measurement(item["top"], TWO_SIGMA * item["height_spread"], "measured", "lidar_camera_ray")
            sill = None
        else:
            height = self.measurement(item["top"] - item["bottom"], TWO_SIGMA * item["height_spread"], "measured", "lidar_camera_ray")
            sill = self.measurement(item["bottom"], TWO_SIGMA * item["height_spread"], "measured", "lidar_camera_ray")

        return Opening(id=opening_id, type=item["type"], wall_id=item["wall_id"], offset_m=round(item["start"], 3),
                       width=width_measurement, height=height, sill_height=sill)

    def build_room(self, room, rooms: list, openings: list, label: str, name: str) -> tuple:
        walls = []
        for edge in room.edges:
            walls.append(Wall(id=edge.wall_id, start=(round(float(edge.start[0]), 3), round(float(edge.start[1]), 3)),
                              end=(round(float(edge.end[0]), 3), round(float(edge.end[1]), 3)),
                              thickness_m=self.wall_thickness(room, edge, rooms), length=self.wall_length(room, edge)))

        edge_by_id = {}
        for edge in room.edges:
            edge_by_id[edge.wall_id] = edge

        counters = {"door": 0, "window": 0}
        room_openings = []
        links = []
        for item in sorted(openings, key=self.start_of):
            if item["room_index"] != room.index:
                continue

            counters[item["type"]] += 1
            prefix = "D" if item["type"] == "door" else "N"
            opening = self.build_opening(f"{prefix}{counters[item['type']]}", item)

            if item["type"] == "door":
                centre = (item["start"] + item["end"]) / 2
                other = RoomFinder(rooms).room_behind(room, edge_by_id[item["wall_id"]], centre, LEADS_TO_REACH_M)
                if other is not None:
                    opening.leads_to = f"R{other.index}"
                    links.append((f"R{other.index}", opening.id))

            room_openings.append(opening)

        half_width = TWO_SIGMA * 0.02 * room.area
        plan_room = Room(id=f"R{room.index}", label=label, name=name, walls=walls, openings=room_openings,
                         ceiling_height=self.room_ceiling(room), floor_area=self.measurement(room.area, half_width))
        return plan_room, links

    def start_of(self, item: dict) -> tuple:
        return item["wall_id"], item["start"]

    def adjacency(self, links_by_room: dict) -> list:
        pairs = []
        seen = set()

        for room_id, links in links_by_room.items():
            for other_id, opening_id in links:
                key = tuple(sorted((room_id, other_id)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(Adjacency(room_a=room_id, room_b=other_id, via_opening=f"{room_id}:{opening_id}"))

        return pairs

    def build(self, rooms: list, openings: list, capture_name: str, room_type: str, extra_notes: list) -> FloorPlan:
        self.notes = []
        plan_rooms = []
        links_by_room = {}

        for room in rooms:
            label = room_type if len(rooms) == 1 else "other"
            plan_room, links = self.build_room(room, rooms, openings, label, f"Room {room.index}")
            plan_rooms.append(plan_room)
            links_by_room[plan_room.id] = links

        notes = [f"{len(rooms)} rooms from the LiDAR point cloud"] + extra_notes + self.notes
        return FloorPlan(capture=CaptureInfo(id=capture_name, tier="lidar", device="iPhone LiDAR (Stray Scanner)"),
                         rooms=plan_rooms, adjacency=self.adjacency(links_by_room), notes="; ".join(notes))
