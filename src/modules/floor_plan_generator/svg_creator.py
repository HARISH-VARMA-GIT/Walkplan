import math
from html import escape

from models.floor_plan import FloorPlan, Measurement, Room, Wall
from utils.geometry import FloorPlanGeometry


PX_PER_M = 100
MARGIN_M = 1.0
TITLE_HEIGHT_PX = 40
DIM_OFFSET_M = 0.35
ROOM_FILLS = ["#EAF2FB", "#EEF7EC", "#FBF1E6", "#F3ECF8", "#E9F6F6", "#FBEFF1"]
DAMAGE_COLOR = "#D9480F"
DIM_COLOR = "#C0392B"
WALL_COLOR = "#2B2B2B"
WINDOW_COLOR = "#3A7BD5"


class Canvas:

    def __init__(self, plan: FloorPlan, geometry: FloorPlanGeometry):
        xs = []
        ys = []

        for room in plan.rooms:
            for x, y in geometry.room_polygon(room):
                xs.append(x)
                ys.append(y)

        self.min_x = min(xs) - MARGIN_M
        self.max_y = max(ys) + MARGIN_M
        self.width = (max(xs) - min(xs) + 2 * MARGIN_M) * PX_PER_M
        self.height = (max(ys) - min(ys) + 2 * MARGIN_M) * PX_PER_M + TITLE_HEIGHT_PX

    def to_screen(self, x: float, y: float) -> tuple:
        screen_x = (x - self.min_x) * PX_PER_M
        screen_y = (self.max_y - y) * PX_PER_M + TITLE_HEIGHT_PX
        return round(screen_x, 1), round(screen_y, 1)


class SvgCreator:

    def __init__(self):
        self.geometry = FloorPlanGeometry()

    def format_length(self, measurement: Measurement, fallback: float = None) -> str:
        if measurement is None:
            if fallback is None:
                return "n/a"
            return f"{fallback:.2f} m"

        half_width = self.geometry.half_width(measurement)

        if half_width >= 0.005:
            return f"{measurement.value:.2f} ±{half_width:.2f} m"

        return f"{measurement.value:.2f} m"

    def wall_direction(self, wall: Wall) -> tuple:
        (x1, y1), (x2, y2) = wall.start, wall.end
        length = math.hypot(x2 - x1, y2 - y1) or 1.0
        return (x2 - x1) / length, (y2 - y1) / length

    def outward_normal(self, room: Room, wall: Wall) -> tuple:
        dx, dy = self.wall_direction(wall)

        if self.geometry.is_counter_clockwise(room):
            return dy, -dx

        return -dy, dx

    def is_point_inside(self, point: tuple, polygon: list) -> bool:
        x, y = point
        inside = False

        for index in range(len(polygon)):
            x1, y1 = polygon[index]
            x2, y2 = polygon[(index + 1) % len(polygon)]

            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside

        return inside

    def point_behind_wall(self, room: Room, wall: Wall) -> tuple:
        out_x, out_y = self.outward_normal(room, wall)
        mid_x = (wall.start[0] + wall.end[0]) / 2
        mid_y = (wall.start[1] + wall.end[1]) / 2
        distance = wall.thickness_m + 0.05
        return mid_x + out_x * distance, mid_y + out_y * distance

    def neighbour_rooms(self, plan: FloorPlan, room: Room, wall: Wall) -> list:
        probe = self.point_behind_wall(room, wall)
        neighbours = []

        for other in plan.rooms:
            if other.id != room.id and self.is_point_inside(probe, self.geometry.room_polygon(other)):
                neighbours.append(other)

        return neighbours

    def polygon_centre(self, room: Room) -> tuple:
        polygon = self.geometry.room_polygon(room)
        centre_x = sum(point[0] for point in polygon) / len(polygon)
        centre_y = sum(point[1] for point in polygon) / len(polygon)
        return centre_x, centre_y

    def draw_room(self, canvas: Canvas, room: Room, index: int) -> str:
        points = []

        for x, y in self.geometry.room_polygon(room):
            screen_x, screen_y = canvas.to_screen(x, y)
            points.append(f"{screen_x},{screen_y}")

        area = self.geometry.room_area(room)
        tip = f"{room.name or room.label} ({room.id}) | area {area:.2f} m² | ceiling {self.format_length(room.ceiling_height)}"
        fill = ROOM_FILLS[index % len(ROOM_FILLS)]

        return (f'<polygon points="{" ".join(points)}" fill="{fill}" '
                f'data-room="{room.id}" data-label="{room.label}"><title>{escape(tip)}</title></polygon>')

    def draw_wall(self, canvas: Canvas, room: Room, wall: Wall) -> str:
        start_x, start_y = canvas.to_screen(*wall.start)
        end_x, end_y = canvas.to_screen(*wall.end)

        out_x, out_y = self.outward_normal(room, wall)
        half = wall.thickness_m / 2 * PX_PER_M
        shift_x = out_x * half
        shift_y = -out_y * half

        length = self.geometry.wall_length(wall)
        label = self.format_length(wall.length, self.geometry.geometric_length(wall))

        return (f'<line x1="{start_x + shift_x}" y1="{start_y + shift_y}" x2="{end_x + shift_x}" y2="{end_y + shift_y}" '
                f'stroke="{WALL_COLOR}" stroke-width="{wall.thickness_m * PX_PER_M}" stroke-linecap="square" '
                f'data-wall="{wall.id}" data-room="{room.id}" data-length="{length:.3f}">'
                f'<title>{wall.id}: {escape(label)}</title></line>')

    def draw_dimension(self, canvas: Canvas, plan: FloorPlan, room: Room, wall: Wall) -> str:
        neighbours = self.neighbour_rooms(plan, room, wall)
        is_shared = len(neighbours) > 0

        if is_shared:
            smallest_id = min([room.id] + [other.id for other in neighbours])
            if room.id != smallest_id:
                return ""

        out_x, out_y = self.outward_normal(room, wall)

        if is_shared:
            offset = -0.3
        else:
            offset = DIM_OFFSET_M + wall.thickness_m

        a = canvas.to_screen(wall.start[0] + out_x * offset, wall.start[1] + out_y * offset)
        b = canvas.to_screen(wall.end[0] + out_x * offset, wall.end[1] + out_y * offset)
        mid_x = (a[0] + b[0]) / 2
        mid_y = (a[1] + b[1]) / 2

        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        if angle > 90 or angle < -90:
            angle += 180

        tick = 6
        tick_x = out_x * tick
        tick_y = -out_y * tick
        label = self.format_length(wall.length, self.geometry.geometric_length(wall))

        return (f'<g data-dim="{wall.id}">'
                f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{DIM_COLOR}" stroke-width="1"/>'
                f'<line x1="{a[0] - tick_x}" y1="{a[1] - tick_y}" x2="{a[0] + tick_x}" y2="{a[1] + tick_y}" stroke="{DIM_COLOR}" stroke-width="1"/>'
                f'<line x1="{b[0] - tick_x}" y1="{b[1] - tick_y}" x2="{b[0] + tick_x}" y2="{b[1] + tick_y}" stroke="{DIM_COLOR}" stroke-width="1"/>'
                f'<text x="{mid_x}" y="{mid_y}" transform="rotate({angle:.1f} {mid_x} {mid_y}) translate(0 -4)" '
                f'text-anchor="middle" font-size="12" fill="{DIM_COLOR}">{escape(label)}</text></g>')

    def draw_opening(self, canvas: Canvas, room: Room, opening, drawn_doors: set) -> str:
        wall = None
        for candidate in room.walls:
            if candidate.id == opening.wall_id:
                wall = candidate

        if wall is None:
            return ""

        dir_x, dir_y = self.wall_direction(wall)
        out_x, out_y = self.outward_normal(room, wall)

        near = (wall.start[0] + dir_x * opening.offset_m, wall.start[1] + dir_y * opening.offset_m)
        far = (near[0] + dir_x * opening.width.value, near[1] + dir_y * opening.width.value)
        to_centre_x = out_x * wall.thickness_m / 2
        to_centre_y = out_y * wall.thickness_m / 2

        gap_start = canvas.to_screen(near[0] + to_centre_x, near[1] + to_centre_y)
        gap_end = canvas.to_screen(far[0] + to_centre_x, far[1] + to_centre_y)

        tip = f"{opening.type} {opening.id}: width {self.format_length(opening.width)}"
        if opening.leads_to:
            tip += f" → {opening.leads_to}"

        parts = [f'<line x1="{gap_start[0]}" y1="{gap_start[1]}" x2="{gap_end[0]}" y2="{gap_end[1]}" '
                 f'stroke="white" stroke-width="{wall.thickness_m * PX_PER_M + 2}"/>']

        door_pair = None
        already_drawn = False
        if opening.leads_to:
            door_pair = frozenset((room.id, opening.leads_to))
            already_drawn = door_pair in drawn_doors

        if opening.type == "door" and door_pair is not None:
            drawn_doors.add(door_pair)

        if opening.type == "door" and not already_drawn:
            parts.extend(self.draw_door_leaf(canvas, room, opening, near, far, out_x, out_y))

        if opening.type == "window":
            for step in (-1, 0, 1):
                shift = step * wall.thickness_m / 3
                a = canvas.to_screen(near[0] + to_centre_x + out_x * shift, near[1] + to_centre_y + out_y * shift)
                b = canvas.to_screen(far[0] + to_centre_x + out_x * shift, far[1] + to_centre_y + out_y * shift)
                parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{WINDOW_COLOR}" stroke-width="1.5"/>')

        return (f'<g data-opening="{opening.id}" data-type="{opening.type}" data-wall="{wall.id}" '
                f'data-width="{opening.width.value:.3f}"><title>{escape(tip)}</title>{"".join(parts)}</g>')

    def draw_door_leaf(self, canvas: Canvas, room: Room, opening, near: tuple, far: tuple, out_x: float, out_y: float) -> list:
        if opening.swing == "left":
            hinge, free = near, far
        else:
            hinge, free = far, near

        radius = opening.width.value
        leaf_end = (hinge[0] - out_x * radius, hinge[1] - out_y * radius)

        hinge_px = canvas.to_screen(*hinge)
        free_px = canvas.to_screen(*free)
        leaf_px = canvas.to_screen(*leaf_end)

        sweep = 1 if (opening.swing == "left") == self.geometry.is_counter_clockwise(room) else 0
        radius_px = radius * PX_PER_M

        return [
            f'<line x1="{hinge_px[0]}" y1="{hinge_px[1]}" x2="{leaf_px[0]}" y2="{leaf_px[1]}" stroke="{WALL_COLOR}" stroke-width="2"/>',
            f'<path d="M {leaf_px[0]} {leaf_px[1]} A {radius_px} {radius_px} 0 0 {sweep} {free_px[0]} {free_px[1]}" '
            f'fill="none" stroke="{WALL_COLOR}" stroke-width="1" stroke-dasharray="4 3"/>',
        ]

    def draw_damage(self, canvas: Canvas, room: Room, damage) -> str:
        tip = (f"{damage.damage_class} on {damage.surface_type} {damage.wall_id or ''}: "
               f"{damage.extent_m2.value:.2f} m² (conf {damage.confidence:.0%})")

        wall = None
        for candidate in room.walls:
            if candidate.id == damage.wall_id:
                wall = candidate

        if damage.surface_type == "wall" and wall is not None:
            out_x, out_y = self.outward_normal(room, wall)
            mid_x = (wall.start[0] + wall.end[0]) / 2
            mid_y = (wall.start[1] + wall.end[1]) / 2
            x, y = canvas.to_screen(mid_x - out_x * 0.15, mid_y - out_y * 0.15)
        else:
            centre_x, centre_y = self.polygon_centre(room)
            x, y = canvas.to_screen(centre_x, centre_y - 0.5)

        return (f'<g data-damage="{damage.id}" data-class="{damage.damage_class}"><title>{escape(tip)}</title>'
                f'<circle cx="{x}" cy="{y}" r="9" fill="{DAMAGE_COLOR}" fill-opacity="0.85"/>'
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="11" fill="white" font-weight="bold">!</text></g>')

    def draw_label(self, canvas: Canvas, room: Room) -> str:
        centre_x, centre_y = self.polygon_centre(room)
        x, y = canvas.to_screen(centre_x, centre_y)

        title = escape(room.name or room.label.replace("_", " ").title())
        area = self.geometry.area_interval(room)
        area_half_width = self.geometry.half_width(area)

        subtitle = f"{area.value:.1f} m²"
        if area_half_width >= 0.05:
            subtitle += f" ±{area_half_width:.1f}"
        if room.ceiling_height is not None:
            subtitle += f" · h {room.ceiling_height.value:.2f} m"

        return (f'<g data-label-for="{room.id}"><text x="{x}" y="{y}" text-anchor="middle" font-size="15" '
                f'font-weight="600" fill="#1F2D3D">{title}</text>'
                f'<text x="{x}" y="{y + 18}" text-anchor="middle" font-size="12" fill="#4A5868">{escape(subtitle)}</text></g>')

    def draw_scale_bar(self, canvas: Canvas) -> str:
        start = (20, canvas.height - 20)
        end = (start[0] + PX_PER_M, start[1])

        return (f'<g id="scale"><line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" stroke="{WALL_COLOR}" stroke-width="3"/>'
                f'<text x="{start[0]}" y="{start[1] - 6}" font-size="11" fill="{WALL_COLOR}">1 m</text></g>')

    def draw_header(self, plan: FloorPlan) -> str:
        total = self.geometry.total_area(plan)
        return (f'<text x="12" y="24" font-size="14" font-weight="600" fill="#1F2D3D">'
                f'{escape(plan.capture.id)} · tier: {plan.capture.tier} · total {total:.1f} m²</text>')

    def create_svg(self, plan: FloorPlan) -> str:
        canvas = Canvas(plan, self.geometry)
        drawn_doors = set()

        rooms = []
        walls = []
        openings = []
        damages = []
        dimensions = []
        labels = []

        for index, room in enumerate(plan.rooms):
            rooms.append(self.draw_room(canvas, room, index))

            for wall in room.walls:
                walls.append(self.draw_wall(canvas, room, wall))
                dimensions.append(self.draw_dimension(canvas, plan, room, wall))

            for opening in room.openings:
                openings.append(self.draw_opening(canvas, room, opening, drawn_doors))

            for damage in room.damage:
                damages.append(self.draw_damage(canvas, room, damage))

            labels.append(self.draw_label(canvas, room))

        body = "\n".join([
            f'<g id="rooms">{"".join(rooms)}</g>',
            f'<g id="walls">{"".join(walls)}</g>',
            f'<g id="openings">{"".join(openings)}</g>',
            f'<g id="damage">{"".join(damages)}</g>',
            f'<g id="dimensions">{"".join(dimensions)}</g>',
            f'<g id="labels">{"".join(labels)}</g>',
            self.draw_scale_bar(canvas),
            self.draw_header(plan),
        ])

        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width:.0f}" height="{canvas.height:.0f}" '
                f'viewBox="0 0 {canvas.width:.0f} {canvas.height:.0f}" font-family="Helvetica, Arial, sans-serif" '
                f'data-units="m" data-px-per-m="{PX_PER_M}">\n'
                f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n')
