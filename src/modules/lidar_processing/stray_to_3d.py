"""Build a 3D mesh + point cloud from a Stray Scanner capture.

Folder layout expected (Stray Scanner export):
  rgb.mp4, depth/000000.png ..., confidence/000000.png ..., camera_matrix.csv, odometry.csv

Install:  pip install open3d opencv-python numpy
Run:      python stray_to_3d.py path/to/capture --every 3
Outputs:  capture/mesh.ply, capture/points.ply, capture/depth_preview.png  (open .ply in MeshLab/CloudCompare or online viewer)

If the mesh looks smeared/duplicated, rerun with --no-flip (camera-axis convention toggle).
"""
import argparse, csv, os
import numpy as np, cv2, open3d as o3d


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def load_poses(path):
    poses = {}
    with open(path) as f:
        rows = csv.reader(f)
        next(rows)
        for r in rows:
            r = [c.strip() for c in r]
            T = np.eye(4)
            T[:3, :3] = quat_to_R(*map(float, r[5:9]))
            T[:3, 3] = list(map(float, r[2:5]))
            poses[int(r[1])] = T  # camera-to-world (ARKit)
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--every", type=int, default=3, help="use every Nth frame")
    ap.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size in metres")
    ap.add_argument("--max-depth", type=float, default=4.0, help="ignore depth beyond this (m)")
    ap.add_argument("--min-conf", type=int, default=2, help="keep pixels with confidence >= this (0,1,2)")
    ap.add_argument("--no-flip", action="store_true", help="don't convert ARKit camera axes to OpenCV")
    a = ap.parse_args()
    d = a.capture

    K_rgb = np.loadtxt(os.path.join(d, "camera_matrix.csv"), delimiter=",")
    poses = load_poses(os.path.join(d, "odometry.csv"))
    depth_files = sorted(f for f in os.listdir(os.path.join(d, "depth")) if f.endswith(".png"))
    video = cv2.VideoCapture(os.path.join(d, "rgb.mp4")) if os.path.exists(os.path.join(d, "rgb.mp4")) else None

    first = cv2.imread(os.path.join(d, "depth", depth_files[0]), cv2.IMREAD_UNCHANGED)
    dh, dw = first.shape
    print(f"{len(depth_files)} depth frames, {dw}x{dh}, dtype={first.dtype}, max raw value={first.max()} (mm)")
    # Save a visible preview: raw depth is 16-bit millimetres, so it looks black in normal viewers.
    prev = cv2.applyColorMap(cv2.convertScaleAbs(first, alpha=255.0 / max(first.max(), 1)), cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(d, "depth_preview.png"), prev)

    # Intrinsics are for the RGB resolution; scale them down to the depth resolution.
    rgb_w = K_rgb[0, 2] * 2
    s = dw / rgb_w
    intr = o3d.camera.PinholeCameraIntrinsic(dw, dh, K_rgb[0, 0] * s, K_rgb[1, 1] * s, K_rgb[0, 2] * s, K_rgb[1, 2] * s)
    flip = np.diag([1, -1, -1, 1]) if not a.no_flip else np.eye(4)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=a.voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    frame_idx, used = -1, 0
    for f in depth_files:
        i = int(os.path.splitext(f)[0])
        rgb = None
        if video is not None:  # advance the video to frame i
            while frame_idx < i:
                ok, rgb = video.read()
                frame_idx += 1
                if not ok:
                    rgb = None
                    break
        if i % a.every or i not in poses:
            continue
        depth = cv2.imread(os.path.join(d, "depth", f), cv2.IMREAD_UNCHANGED)
        cpath = os.path.join(d, "confidence", f)
        if os.path.exists(cpath):
            conf = cv2.imread(cpath, cv2.IMREAD_UNCHANGED)
            depth = np.where(conf >= a.min_conf, depth, 0).astype(np.uint16)
        color = cv2.cvtColor(cv2.resize(rgb, (dw, dh)), cv2.COLOR_BGR2RGB) if rgb is not None \
            else np.full((dh, dw, 3), 180, np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(depth),
            depth_scale=1000.0, depth_trunc=a.max_depth, convert_rgb_to_intensity=False)
        extrinsic = np.linalg.inv(poses[i] @ flip)  # world-to-camera
        vol.integrate(rgbd, intr, extrinsic)
        used += 1

    print(f"integrated {used} frames")
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = vol.extract_point_cloud()
    o3d.io.write_triangle_mesh(os.path.join(d, "mesh.ply"), mesh)
    o3d.io.write_point_cloud(os.path.join(d, "points.ply"), pcd)
    bb = pcd.get_axis_aligned_bounding_box().get_extent()
    print(f"bounding box (m): {bb[0]:.2f} x {bb[1]:.2f} x {bb[2]:.2f}  (y is up in ARKit world)")
    print("wrote mesh.ply, points.ply, depth_preview.png")


if __name__ == "__main__":
    main()
