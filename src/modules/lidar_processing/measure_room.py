"""Measure a single room from a fused point cloud: floor, ceiling height, wall outline, wall lengths, area.

Run:  uv run measure_room.py ../data/single_room/c00a170fe1
Needs: <capture>/points.ply (from stray_to_3d.py) and <capture>/odometry.csv
Writes: <capture>/room.json and <capture>/room_plan.png

Method (ARKit world: y is up, gravity-aligned):
  1. Floor / ceiling: height histogram of horizontal surfaces; floor = lowest strong peak,
     ceiling = highest strong peak above the camera path (reported missing if never seen).
  2. Wall direction: dominant wall-normal angle (mod 90 deg) -> rotate so walls are axis-aligned.
  3. Wall lines: 1-D peaks of wall-point positions along each axis (inner wall surfaces).
  4. Room shape: the wall lines cut the floor into grid cells; a cell is inside if it is well covered
     by observed (non-wall) points; keep the connected group of cells containing the camera path.
     Handles rectangles and L-shapes; door leakage is limited because cells beyond a door are poorly covered.
  5. Outline -> polygon -> wall lengths + area. Sigma per wall = spread of that wall's points.
"""
import argparse, csv, json, os
import numpy as np, cv2, open3d as o3d


def peaks_1d(vals, bin_m=0.01, min_count=100, rel=0.05, merge_m=0.06):
    if len(vals) == 0:
        return []
    lo, hi = vals.min() - 0.05, vals.max() + 0.05
    hist, edges = np.histogram(vals, bins=max(int((hi - lo) / bin_m), 1), range=(lo, hi))
    sm = np.convolve(hist, [1, 2, 3, 2, 1], mode="same") / 9.0
    idx = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]
    if not idx:
        return []
    top = max(sm[i] for i in idx)
    out = []
    for i in sorted(idx, key=lambda i: -sm[i]):
        c = (edges[i] + edges[i + 1]) / 2
        near = vals[np.abs(vals - c) < 0.03]
        if len(near) < min_count or sm[i] < rel * top:
            continue
        c = float(np.median(near))
        if any(abs(c - p["pos"]) < merge_m for p in out):
            continue
        out.append({"pos": c, "count": int(len(near)), "std": float(np.std(near))})
    return sorted(out, key=lambda p: p["pos"])


def load_traj(cap):
    with open(os.path.join(cap, "odometry.csv")) as f:
        rows = csv.reader(f); next(rows)
        return np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--cloud", default="points.ply")
    a = ap.parse_args()
    cap = a.capture

    pcd = o3d.io.read_point_cloud(os.path.join(cap, a.cloud)).voxel_down_sample(0.02)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    P, N = np.asarray(pcd.points), np.asarray(pcd.normals)
    traj = load_traj(cap)
    print(f"{len(P)} points after 2 cm downsampling, {len(traj)} camera poses")

    # 1. floor & ceiling
    horiz = np.abs(N[:, 1]) > 0.9
    hp = peaks_1d(P[horiz, 1], min_count=200, rel=0.15)
    if not hp:
        raise SystemExit("no horizontal surfaces found")
    floor = hp[0]
    cam_top = np.percentile(traj[:, 1], 95)
    ceil_cands = [p for p in hp if p["pos"] > cam_top + 0.1 and p["pos"] - floor["pos"] > 1.8]
    ceiling = ceil_cands[-1] if ceil_cands else None
    cam_h = float(np.median(traj[:, 1]) - floor["pos"])
    print(f"floor at y={floor['pos']:.3f} (sigma {floor['std']*100:.1f} cm); camera ~{cam_h:.2f} m above floor")
    ch = None
    if ceiling:
        ch = ceiling["pos"] - floor["pos"]
        ch_sigma = float(np.hypot(floor["std"], ceiling["std"]) / np.sqrt(min(floor["count"], ceiling["count"])) + 0.005)
        print(f"ceiling height {ch:.3f} m")
    else:
        print("ceiling not observed in this capture -> ceiling height not reported")

    # 2. dominant wall direction
    top = ceiling["pos"] - 0.15 if ceiling else floor["pos"] + 2.2
    wall = (np.abs(N[:, 1]) < 0.15) & (P[:, 1] > floor["pos"] + 0.15) & (P[:, 1] < top)
    ang = np.degrees(np.arctan2(N[wall, 2], N[wall, 0])) % 90.0
    h, e = np.histogram(ang, bins=180, range=(0, 90))
    h = np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(5), "same")[3:-3]  # circular smoothing
    theta = np.radians((e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2)
    c, s = np.cos(-theta), np.sin(-theta)
    rot = np.array([[c, -s], [s, c]])
    XY = P[:, [0, 2]] @ rot.T
    NXY = N[:, [0, 2]] @ rot.T
    T2 = traj[:, [0, 2]] @ rot.T

    # 3. wall lines along each axis
    lines = []
    for ax in (0, 1):
        m = wall & (np.abs(NXY[:, ax]) > 0.9)
        lines.append(peaks_1d(XY[m, ax], min_count=60, rel=0.05))
    if len(lines[0]) < 2 or len(lines[1]) < 2:
        raise SystemExit(f"not enough wall lines found: {len(lines[0])} x, {len(lines[1])} z")
    xs, zs = [p["pos"] for p in lines[0]], [p["pos"] for p in lines[1]]

    # 4. cell coverage by observed non-wall points (floor, furniture, anything below ceiling)
    body = (~wall) & (P[:, 1] > floor["pos"] - 0.05) & (P[:, 1] < (ceiling["pos"] - 0.05 if ceiling else top))
    g = 0.05
    occ = set(map(tuple, np.floor(XY[body] / g).astype(int)))
    cov = np.zeros((len(xs) - 1, len(zs) - 1))
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            gx = np.arange(np.floor(xs[i] / g) + 1, np.floor(xs[i + 1] / g)).astype(int)
            gz = np.arange(np.floor(zs[j] / g) + 1, np.floor(zs[j + 1] / g)).astype(int)
            n = len(gx) * len(gz)
            cov[i, j] = sum((x, z) in occ for x in gx for z in gz) / n if n else 0
    inside = cov >= 0.5
    seeds = {(int(np.searchsorted(xs, x) - 1), int(np.searchsorted(zs, z) - 1)) for x, z in T2}
    seeds = {(i, j) for i, j in seeds if 0 <= i < len(xs) - 1 and 0 <= j < len(zs) - 1}
    keep, stack = np.zeros_like(inside), [sd for sd in seeds if inside[sd]]
    while stack:
        i, j = stack.pop()
        if keep[i, j]:
            continue
        keep[i, j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < len(xs) - 1 and 0 <= nj < len(zs) - 1 and inside[ni, nj] and not keep[ni, nj]:
                stack.append((ni, nj))
    if not keep.any():
        raise SystemExit("room region not found (camera path not inside a covered cell)")

    # 5. outline polygon from kept cells (1 cm raster)
    x0, z0 = xs[0], zs[0]
    W, H = int((xs[-1] - x0) / 0.01) + 3, int((zs[-1] - z0) / 0.01) + 3
    mask = np.zeros((H, W), np.uint8)
    for i, j in zip(*np.nonzero(keep)):
        cv2.rectangle(mask, (int((xs[i] - x0) / 0.01) + 1, int((zs[j] - z0) / 0.01) + 1),
                      (int((xs[i + 1] - x0) / 0.01), int((zs[j + 1] - z0) / 0.01)), 255, -1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(cnt, 3, True)[:, 0, :].astype(float)
    snap = lambda v, L: min(L, key=lambda l: abs(l - v))
    verts = [(snap(px * 0.01 + x0, xs), snap(pz * 0.01 + z0, zs)) for px, pz in poly]
    verts = [v for k, v in enumerate(verts) if v != verts[k - 1]]
    sig = {("x", p["pos"]): p["std"] / np.sqrt(p["count"]) + 0.005 for p in lines[0]}
    sig.update({("z", p["pos"]): p["std"] / np.sqrt(p["count"]) + 0.005 for p in lines[1]})
    walls = []
    for k in range(len(verts)):
        (ax_, az_), (bx_, bz_) = verts[k], verts[(k + 1) % len(verts)]
        L = float(np.hypot(bx_ - ax_, bz_ - az_))
        # wall length is bounded by the two perpendicular walls at its ends
        if abs(ax_ - bx_) < 1e-9:   # runs along z, ends on z-lines
            sd = float(np.hypot(sig[("z", az_)], sig[("z", bz_)]))
        else:
            sd = float(np.hypot(sig[("x", ax_)], sig[("x", bx_)]))
        walls.append({"id": f"W{k+1}", "length_m": round(L, 3), "sigma_m": round(sd, 4),
                      "start": [round(ax_, 3), round(az_, 3)], "end": [round(bx_, 3), round(bz_, 3)]})
    vx, vz = np.array(verts).T
    area = float(abs(np.dot(vx, np.roll(vz, -1)) - np.dot(vz, np.roll(vx, -1))) / 2)

    out = {
        "capture": os.path.basename(os.path.normpath(cap)),
        "frame": "room-aligned: ARKit world rotated about y by %.2f deg" % np.degrees(theta),
        "floor_y_m": round(floor["pos"], 4),
        "ceiling_height_m": round(ch, 3) if ch else None,
        "ceiling_height_sigma_m": round(ch_sigma, 4) if ch else None,
        "ceiling_note": None if ch else "ceiling not observed in capture",
        "floor_area_m2": round(area, 2),
        "polygon_m": [[round(x, 3), round(z, 3)] for x, z in verts],
        "walls": walls,
        "debug": {"x_lines": lines[0], "z_lines": lines[1], "cell_coverage": cov.round(2).tolist()},
    }
    with open(os.path.join(cap, "room.json"), "w") as f:
        json.dump(out, f, indent=2)

    # render plan
    sc, pad = 120, 80
    img = np.full((int((zs[-1] - z0) * sc) + 2 * pad, int((xs[-1] - x0) * sc) + 2 * pad, 3), 255, np.uint8)
    to_px = lambda x, z: (int((x - x0) * sc) + pad, int((z - z0) * sc) + pad)
    occ_pts = XY[body][::5]
    for x, z in occ_pts:
        if x0 <= x <= xs[-1] and z0 <= z <= zs[-1]:
            cv2.circle(img, to_px(x, z), 1, (215, 215, 215), -1)
    cv2.polylines(img, [np.array([to_px(x, z) for x, z in verts])], True, (40, 40, 40), 3)
    for w in walls:
        (sx, sz), (ex, ez) = w["start"], w["end"]
        mx, mz = to_px((sx + ex) / 2, (sz + ez) / 2)
        cv2.putText(img, f"{w['length_m']:.2f} m", (mx - 35, mz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    t = to_px(*T2.T.mean(axis=1))
    label = f"area {area:.2f} m2" + (f" | ceiling {ch:.2f} m" if ch else " | ceiling n/a")
    cv2.putText(img, label, (t[0] - 90, t[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1)
    cv2.polylines(img, [np.array([to_px(x, z) for x, z in T2[::10]])], False, (230, 160, 60), 1)
    cv2.imwrite(os.path.join(cap, "room_plan.png"), img)

    print(f"floor area {area:.2f} m2, {len(walls)} walls:")
    for w in walls:
        print(f"  {w['id']}: {w['length_m']:.3f} m  +/- {w['sigma_m']*100:.1f} cm")
    print("wrote room.json, room_plan.png")


if __name__ == "__main__":
    main()
