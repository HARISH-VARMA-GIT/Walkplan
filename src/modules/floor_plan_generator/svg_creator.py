"""Render a FloorPlan to SVG (no dependencies beyond the schema).

Layers (draw order, OAS-style):  rooms -> walls -> openings -> damage -> dimensions -> labels
Every element carries data-* attributes and a <title> tooltip, so the SVG stays tied to the JSON.

CLI:  python -m tapeless.cli render plan.json [--png plan.png]
"""
from __future__ import annotations

import argparse
import math
from html import escape

from models.floor_plan import FloorPlan, Measurement, Room, Wall

PX_PER_M = 100
MARGIN_M = 1.0
DIM_OFFSET_M = 0.35
ROOM_FILLS = ["#EAF2FB", "#EEF7EC", "#FBF1E6", "#F3ECF8", "#E9F6F6", "#FBEFF1"]
DAMAGE_COLOR = "#D9480F"


def _fmt(m: Measurement | None, fallback: float | None = None) -> str:
    if m is None:
        return f"{fallback:.2f} m" if fallback is not None else "n/a"
    return f"{m.value:.2f} ±{m.half_width:.2f} m" if m.half_width >= 0.005 else f"{m.value:.2f} m"


class _Canvas:
    def __init__(self, plan: FloorPlan):
        xs = [p[0] for r in plan.rooms for p in r.polygon]
        ys = [p[1] for r in plan.rooms for p in r.polygon]
        self.minx, self.maxy = min(xs) - MARGIN_M, max(ys) + MARGIN_M
        self.w = (max(xs) - min(xs) + 2 * MARGIN_M) * PX_PER_M
        self.h = (max(ys) - min(ys) + 2 * MARGIN_M) * PX_PER_M + 40  # + title strip

    def pt(self, x: float, y: float) -> tuple[float, float]:  # plan (y up) -> screen (y down)
        return round((x - self.minx) * PX_PER_M, 1), round((self.maxy - y) * PX_PER_M + 40, 1)


def _unit(w: Wall):
    (x1, y1), (x2, y2) = w.start, w.end
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    return (x2 - x1) / L, (y2 - y1) / L


def _outward(room: Room, w: Wall):
    ux, uy = _unit(w)
    # for a CCW room the interior is on the left, so outward is the right-hand normal
    return (uy, -ux) if room.is_ccw else (-uy, ux)


def _inside(pt, poly) -> bool:
    x, y = pt; ins = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            ins = not ins
    return ins


def _is_shared(plan: FloorPlan, room: Room, w: Wall) -> bool:
    """True if just beyond this wall's outer face lies another room (an interior partition)."""
    ox, oy = _outward(room, w)
    mx, my = (w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2
    probe = (mx + ox * (w.thickness_m + 0.05), my + oy * (w.thickness_m + 0.05))
    return any(r.id != room.id and _inside(probe, r.polygon) for r in plan.rooms)


def _shares(other: Room, w: Wall, room: Room) -> bool:
    if other.id == room.id:
        return True
    ox, oy = _outward(room, w)
    mx, my = (w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2
    return _inside((mx + ox * (w.thickness_m + 0.05), my + oy * (w.thickness_m + 0.05)), other.polygon)


def render_svg(plan: FloorPlan) -> str:
    c = _Canvas(plan)
    drawn_doors: set[frozenset] = set()  # a door between R1 and R2 is listed in both rooms; draw its leaf once
    rooms, walls, openings, damage, dims, labels = [], [], [], [], [], []

    for k, r in enumerate(plan.rooms):
        pts = " ".join(f"{x},{y}" for x, y in (c.pt(*p) for p in r.polygon))
        tip = f"{r.name or r.label} ({r.id}) | area {r.geometric_area:.2f} m² | ceiling {_fmt(r.ceiling_height)}"
        rooms.append(f'<polygon points="{pts}" fill="{ROOM_FILLS[k % len(ROOM_FILLS)]}" '
                     f'data-room="{r.id}" data-label="{r.label}"><title>{escape(tip)}</title></polygon>')

        for w in r.walls:
            (sx, sy), (ex, ey) = c.pt(*w.start), c.pt(*w.end)
            # shift the stroke outward by half the thickness so the inner face sits on the coordinates
            ox, oy = _outward(r, w)
            half = w.thickness_m / 2 * PX_PER_M
            dx, dy = ox * half, -oy * half
            L = w.length.value if w.length else w.geometric_length
            walls.append(f'<line x1="{sx+dx}" y1="{sy+dy}" x2="{ex+dx}" y2="{ey+dy}" stroke="#2B2B2B" '
                         f'stroke-width="{w.thickness_m*PX_PER_M}" stroke-linecap="square" '
                         f'data-wall="{w.id}" data-room="{r.id}" data-length="{L:.3f}">'
                         f'<title>{w.id}: {escape(_fmt(w.length, w.geometric_length))}</title></line>')

            # dimension line: outside for exterior walls, inside the room for shared partitions
            shared = _is_shared(plan, r, w)
            if shared and r.id != min(o_.id for o_ in plan.rooms if _shares(o_, w, r)):
                continue  # partition already dimensioned from the neighbouring room
            off = -0.3 if shared else (DIM_OFFSET_M + w.thickness_m)
            a = c.pt(w.start[0] + ox * off, w.start[1] + oy * off)
            b = c.pt(w.end[0] + ox * off, w.end[1] + oy * off)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            if ang > 90 or ang < -90:
                ang += 180
            tick = 6
            nx, ny = ox * tick, -oy * tick
            dims.append(
                f'<g data-dim="{w.id}"><line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#C0392B" stroke-width="1"/>'
                f'<line x1="{a[0]-nx}" y1="{a[1]-ny}" x2="{a[0]+nx}" y2="{a[1]+ny}" stroke="#C0392B" stroke-width="1"/>'
                f'<line x1="{b[0]-nx}" y1="{b[1]-ny}" x2="{b[0]+nx}" y2="{b[1]+ny}" stroke="#C0392B" stroke-width="1"/>'
                f'<text x="{mx}" y="{my}" transform="rotate({ang:.1f} {mx} {my}) translate(0 -4)" '
                f'text-anchor="middle" font-size="12" fill="#C0392B">{escape(_fmt(w.length, w.geometric_length))}</text></g>')

        wall_by_id = {w.id: w for w in r.walls}
        for o in r.openings:
            w = wall_by_id[o.wall_id]
            ux, uy = _unit(w)
            ox, oy = _outward(r, w)
            p0 = (w.start[0] + ux * o.offset_m, w.start[1] + uy * o.offset_m)
            p1 = (p0[0] + ux * o.width.value, p0[1] + uy * o.width.value)
            mid = (-ox * w.thickness_m / 2, -oy * w.thickness_m / 2)  # centre of wall thickness
            q0 = c.pt(p0[0] - mid[0], p0[1] - mid[1])
            q1 = c.pt(p1[0] - mid[0], p1[1] - mid[1])
            tip = f"{o.type} {o.id}: width {_fmt(o.width)}" + (f" → {o.leads_to}" if o.leads_to else "")
            parts = [f'<line x1="{q0[0]}" y1="{q0[1]}" x2="{q1[0]}" y2="{q1[1]}" stroke="white" '
                     f'stroke-width="{w.thickness_m*PX_PER_M+2}"/>']  # cut the gap
            pair = frozenset((r.id, o.leads_to)) if o.leads_to else None
            already = pair in drawn_doors if pair else False
            if pair and o.type == "door":
                drawn_doors.add(pair)
            if o.type == "door" and not already:
                hinge, free = (p0, p1) if o.swing == "left" else (p1, p0)
                rad = o.width.value
                leaf_end = (hinge[0] - ox * rad, hinge[1] - oy * rad)  # leaf opens into the room
                h, f_, le = c.pt(*hinge), c.pt(*free), c.pt(*leaf_end)
                sweep = 1 if (o.swing == "left") == r.is_ccw else 0
                parts.append(f'<line x1="{h[0]}" y1="{h[1]}" x2="{le[0]}" y2="{le[1]}" stroke="#2B2B2B" stroke-width="2"/>')
                parts.append(f'<path d="M {le[0]} {le[1]} A {rad*PX_PER_M} {rad*PX_PER_M} 0 0 {sweep} {f_[0]} {f_[1]}" '
                             f'fill="none" stroke="#2B2B2B" stroke-width="1" stroke-dasharray="4 3"/>')
            elif o.type == "window":
                for s in (-1, 0, 1):
                    sh = s * w.thickness_m / 3
                    a = c.pt(p0[0] - mid[0] + ox * sh, p0[1] - mid[1] + oy * sh)
                    b = c.pt(p1[0] - mid[0] + ox * sh, p1[1] - mid[1] + oy * sh)
                    parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#3A7BD5" stroke-width="1.5"/>')
            openings.append(f'<g data-opening="{o.id}" data-type="{o.type}" data-wall="{w.id}" '
                            f'data-width="{o.width.value:.3f}"><title>{escape(tip)}</title>{"".join(parts)}</g>')

        for d in r.damage:
            tip = f"{d.damage_class} on {d.surface_type} {d.wall_id or ''}: {d.extent_m2.value:.2f} m² (conf {d.confidence:.0%})"
            if d.surface_type == "wall" and d.wall_id:
                w = wall_by_id[d.wall_id]
                mx, my = (w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2
                ox, oy = _outward(r, w)
                x, y = c.pt(mx - ox * 0.15, my - oy * 0.15)
            else:
                cx = sum(p[0] for p in r.polygon) / len(r.polygon)
                cy = sum(p[1] for p in r.polygon) / len(r.polygon)
                x, y = c.pt(cx, cy - 0.5)
            damage.append(f'<g data-damage="{d.id}" data-class="{d.damage_class}"><title>{escape(tip)}</title>'
                          f'<circle cx="{x}" cy="{y}" r="9" fill="{DAMAGE_COLOR}" fill-opacity="0.85"/>'
                          f'<text x="{x}" y="{y+4}" text-anchor="middle" font-size="11" fill="white" font-weight="bold">!</text></g>')

        cx = sum(p[0] for p in r.polygon) / len(r.polygon)
        cy = sum(p[1] for p in r.polygon) / len(r.polygon)
        x, y = c.pt(cx, cy)
        title = escape(r.name or r.label.replace("_", " ").title())
        ai = r.area_interval()
        sub = f"{ai.value:.1f} m²" + (f" ±{ai.half_width:.1f}" if ai.half_width >= 0.05 else "") + (f" · h {r.ceiling_height.value:.2f} m" if r.ceiling_height else "")
        labels.append(f'<g data-label-for="{r.id}"><text x="{x}" y="{y}" text-anchor="middle" font-size="15" '
                      f'font-weight="600" fill="#1F2D3D">{title}</text>'
                      f'<text x="{x}" y="{y+18}" text-anchor="middle" font-size="12" fill="#4A5868">{escape(sub)}</text></g>')

    # scale bar + header
    sb0, sb1 = c.pt(c.minx + 0.3, c.maxy - (c.h - 40) / PX_PER_M + 0.3), None
    sb1 = (sb0[0] + PX_PER_M, sb0[1])
    header = (f'<text x="12" y="24" font-size="14" font-weight="600" fill="#1F2D3D">'
              f'{escape(plan.capture.id)} · tier: {plan.capture.tier} · total {plan.total_area():.1f} m²</text>')
    scalebar = (f'<g id="scale"><line x1="{sb0[0]}" y1="{sb0[1]}" x2="{sb1[0]}" y2="{sb1[1]}" stroke="#2B2B2B" stroke-width="3"/>'
                f'<text x="{sb0[0]}" y="{sb0[1]-6}" font-size="11" fill="#2B2B2B">1 m</text></g>')

    body = "\n".join([
        f'<g id="rooms">{"".join(rooms)}</g>',
        f'<g id="walls">{"".join(walls)}</g>',
        f'<g id="openings">{"".join(openings)}</g>',
        f'<g id="damage">{"".join(damage)}</g>',
        f'<g id="dimensions">{"".join(dims)}</g>',
        f'<g id="labels">{"".join(labels)}</g>',
        scalebar, header,
    ])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{c.w:.0f}" height="{c.h:.0f}" '
            f'viewBox="0 0 {c.w:.0f} {c.h:.0f}" font-family="Helvetica, Arial, sans-serif" '
            f'data-schema-version="{plan.schema_version}" data-units="m" data-px-per-m="{PX_PER_M}">\n'
            f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n')