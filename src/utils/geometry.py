import math

from models.floor_plan import FloorPlan, Measurement, Room, Wall


class FloorPlanGeometry:

    def room_polygon(self, room: Room) -> list:
        return [wall.start for wall in room.walls]

    def signed_area(self, polygon: list) -> float:
        total = 0.0

        for index in range(len(polygon)):
            x1, y1 = polygon[index]
            x2, y2 = polygon[(index + 1) % len(polygon)]
            total += x1 * y2 - x2 * y1

        return total / 2

    def is_counter_clockwise(self, room: Room) -> bool:
        return self.signed_area(self.room_polygon(room)) > 0

    def room_area(self, room: Room) -> float:
        return abs(self.signed_area(self.room_polygon(room)))

    def geometric_length(self, wall: Wall) -> float:
        (x1, y1), (x2, y2) = wall.start, wall.end
        return math.hypot(x2 - x1, y2 - y1)

    def wall_length(self, wall: Wall) -> float:
        if wall.length is not None:
            return wall.length.value

        return self.geometric_length(wall)

    def half_width(self, measurement: Measurement) -> float:
        return (measurement.high - measurement.low) / 2

    def relative_error(self, measurement: Measurement) -> float:
        if measurement.value <= 0:
            return 0.0

        return self.half_width(measurement) / measurement.value

    def area_interval(self, room: Room) -> Measurement:
        if room.floor_area is not None:
            return room.floor_area

        area = self.room_area(room)
        errors = []

        for wall in room.walls:
            if wall.length is not None:
                errors.append(self.relative_error(wall.length))

        if not errors:
            return Measurement(value=area, low=area, high=area, method="measured", source="polygon")

        mean_error = sum(errors) / len(errors)
        low = area * (1 - mean_error) ** 2
        high = area * (1 + mean_error) ** 2

        return Measurement(value=area, low=low, high=high, method="estimated", source="polygon")

    def total_area(self, plan: FloorPlan) -> float:
        total = 0.0

        for room in plan.rooms:
            total += self.area_interval(room).value

        return total
