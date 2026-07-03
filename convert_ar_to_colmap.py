"""
将 AR 输出（点云 PLY + 位姿 PLY + 内参 TXT）转换为 COLMAP 格式。

功能模块:
  1. 数据读取:    read_ar_data()
  2. 地面对齐:    fit_ground_plane() + align_scene_to_plane()
  3. 格式转换:    convert_poses_to_colmap()
  4. 格式写出:    write_colmap_model()
  5. 可视化:      visualize()
  6. 内参更新:    update_camera_intrinsics_from_image()
"""

import glob
import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
from PIL import Image


# ==============================================================================
#  工具函数
# ==============================================================================

def quat_to_rot(q):
    """四元数 (qw, qx, qy, qz) → 3×3 旋转矩阵"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ])


def rot_to_quat(R):
    """3×3 旋转矩阵 → 四元数 (qw, qx, qy, qz)"""
    R = np.asarray(R, dtype=np.float64)
    q = np.zeros(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s

    q /= np.linalg.norm(q)
    return q[0], q[1], q[2], q[3]


def _rotation_from_two_vectors(src, dst):
    """计算将单位向量 src 旋转到 dst 的 3×3 旋转矩阵（Rodrigues）"""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)

    if np.allclose(src, dst):
        return np.eye(3)
    if np.allclose(src, -dst):
        # 180° 旋转：绕任意垂直轴
        perp = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(perp, src)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        cos_a, sin_a = -1.0, 0.0
    else:
        axis = np.cross(src, dst)
        axis /= np.linalg.norm(axis)
        cos_a = np.dot(src, dst)
        sin_a = np.linalg.norm(np.cross(src, dst))

    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0],
    ])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


# ==============================================================================
#  1. 数据读取
# ==============================================================================

def _load_point_cloud(ply_path):
    """读取点云 PLY 文件，返回 open3d PointCloud"""
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[INFO] Loaded point cloud: {len(pcd.points)} points")
    pcd.paint_uniform_color([0, 1, 0])
    return pcd


def _load_poses(ply_path):
    """读取位姿 PLY 文件（自定义解析），返回 pose 列表"""
    poses = []
    with open(ply_path, "r") as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break

    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        x, y, z = map(float, parts[0:3])
        qx, qy, qz, qw = map(float, parts[3:7])
        name = parts[7]
        poses.append({
            "t": np.array([x, y, z]),
            "q": np.array([qw, qx, qy, qz]),
            "name": name,
        })

    print(f"[INFO] Loaded poses: {len(poses)} cameras")
    return poses


def _load_intrinsics(txt_path):
    """读取内参文本文件，返回参数字典"""
    params = {}
    with open(txt_path, "r") as f:
        for line in f:
            k, v = line.strip().split("=")
            params[k] = float(v)
    return params


def read_ar_data(input_dir):
    """
    从 AR 输出目录读取全部数据。

    参数:
        input_dir: 包含 scan_all_pointcloud.ply, scan_all_pose.ply,
                   scan_camera_intrinsics.txt 的目录

    返回:
        pcd:         open3d PointCloud
        poses:       list[dict], 每个 dict 含 t, q, name
        intrinsics:  dict 或 None（无内参文件时）
    """
    points_path = os.path.join(input_dir, "scan_all_pointcloud.ply")
    poses_path = os.path.join(input_dir, "scan_all_pose.ply")
    intrinsic_path = os.path.join(input_dir, "scan_camera_intrinsics.txt")

    pcd = _load_point_cloud(points_path)
    poses = _load_poses(poses_path)

    if os.path.exists(intrinsic_path):
        intrinsics = _load_intrinsics(intrinsic_path)
    else:
        intrinsics = None

    return pcd, poses, intrinsics


# ==============================================================================
#  2. 地面对齐：平面拟合 + 场景旋转
# ==============================================================================

def fit_ground_plane(camera_positions, point_cloud_points):
    """
    用 SVD 对相机位姿坐标拟合平面，并用点云确定法线方向（大部分点位于平面下方）。

    参数:
        camera_positions:  (M, 3) numpy array — 相机光心坐标
        point_cloud_points: (N, 3) numpy array — 用于判断法线方向

    返回:
        normal:   (3,) 单位法向量，指向大部分点云的对侧（上方）
        centroid: (3,) 相机位置质心
        offset:   float, 平面方程 normal·p + offset = 0 中的 offset
    """
    cam_pts = np.asarray(camera_positions)

    # SVD 拟合平面：最小奇异值对应的右奇异向量即为法线
    centroid = cam_pts.mean(axis=0)
    centered = cam_pts - centroid
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # 最小奇异值对应的方向

    # 确定法线方向：大部分点云应位于平面下方 (normal·p + offset < 0)
    offset = -np.dot(normal, centroid)
    cloud_pts = np.asarray(point_cloud_points)
    signed_dist = cloud_pts @ normal + offset
    above = int(np.sum(signed_dist > 0))
    below = int(np.sum(signed_dist < 0))

    if above > below:
        normal = -normal
        offset = -offset

    print(f"[INFO] Plane fitted from {len(cam_pts)} camera positions: "
          f"normal={np.round(normal, 4)}, "
          f"cloud_points above={above}, below={below} "
          f"({'flipped' if above > below else 'kept'})")

    return normal, centroid, offset


def align_scene_to_plane(pcd, poses, normal):
    """
    将场景（点云 + 所有相机位姿）旋转，使地平面法线与世界 Z 轴对齐。

    参数:
        pcd:    open3d PointCloud
        poses:  list[dict], 每个含 t, q, name
        normal: (3,) 地平面法向量（单位向量）

    返回:
        pcd:    旋转后的点云
        poses:  旋转后的位姿列表
        R_align: 使用的 3×3 旋转矩阵
    """
    target = np.array([0.0, 0.0, 1.0])
    R_align = _rotation_from_two_vectors(normal, target)

    # 旋转点云
    pts = np.asarray(pcd.points)
    pts_rotated = (R_align @ pts.T).T
    pcd.points = o3d.utility.Vector3dVector(pts_rotated)

    # 旋转相机位姿
    for pose in poses:
        R_old = quat_to_rot(pose["q"])
        t_old = pose["t"]

        R_new = R_align @ R_old
        t_new = R_align @ t_old

        pose["t"] = t_new
        qw, qx, qy, qz = rot_to_quat(R_new)
        pose["q"] = np.array([qw, qx, qy, qz])

    print(f"[INFO] Scene aligned: normal {normal} → Z-axis")
    return pcd, poses, R_align


# ==============================================================================
#  3. 格式转换：AR → COLMAP
# ==============================================================================

def convert_poses_to_colmap(poses, intrinsics, vertical=False, camera_model="SIMPLE_RADIAL"):
    """
    将 AR 位姿 (c2w) 转换为 COLMAP 格式 (w2c)，并生成 camera/image 记录。

    参数:
        poses:        位姿列表（来自 _load_poses）
        intrinsics:   内参字典（来自 _load_intrinsics）或 None
        vertical:     是否竖屏拍摄（额外绕 Z 轴旋转 90°）
        camera_model: COLMAP 相机模型名

    返回:
        cameras: list[dict]  相机记录
        images:  list[dict]  图像记录
    """
    # --- 构建相机记录 ---
    cameras = []
    if intrinsics is not None:
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        w = int(intrinsics["width"])
        h = int(intrinsics["height"])
        cx = w / 2
        cy = h / 2

        # 根据模型确定参数列表
        model_params_map = {
            "PINHOLE":        [fx, fy, cx, cy],
            "SIMPLE_PINHOLE": [max(fx, fy), cx, cy],
            "RADIAL":         [max(fx, fy), cx, cy, 0.0, 0.0],
            "SIMPLE_RADIAL":  [max(fx, fy), cx, cy, 0.0],
        }
        params = model_params_map.get(camera_model, [fx, fy, cx, cy])
        cameras.append({
            "camera_id": 1,
            "model":     camera_model if camera_model in model_params_map else "PINHOLE",
            "width":     w,
            "height":    h,
            "params":    params,
        })

    # --- 构建图像记录 ---
    images = []
    for i, pose in enumerate(poses):
        R = quat_to_rot(pose["q"])
        t = pose["t"]

        # 竖屏修正：绕相机 Z 轴旋转 90°
        if vertical:
            theta = np.radians(90)
            c, s = np.cos(theta), np.sin(theta)
            R_z90 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            R = R @ R_z90

        # c2w → w2c
        R_w2c = R.T
        t_w2c = -R_w2c @ t

        # 绕 X 轴 180° 翻转（AR 相机前向 -Z → COLMAP 前向 +Z）
        R_fix = np.diag([1.0, -1.0, -1.0])
        R_w2c = R_fix @ R_w2c
        t_w2c = R_fix @ t_w2c

        qw, qx, qy, qz = rot_to_quat(R_w2c)

        images.append({
            "image_id":  i + 1,
            "qw":        qw,
            "qx":        qx,
            "qy":        qy,
            "qz":        qz,
            "tx":        t_w2c[0],
            "ty":        t_w2c[1],
            "tz":        t_w2c[2],
            "camera_id": 1,
            "name":      pose["name"],
        })

    return cameras, images


# ==============================================================================
#  4. 写出 COLMAP 格式
# ==============================================================================

def write_colmap_model(cameras, images, points3d, output_dir, fmt="txt"):
    """
    将 COLMAP 数据写入磁盘（cameras.txt, images.txt, points3D.txt）。

    参数:
        cameras:   list[dict]  相机记录
        images:    list[dict]  图像记录
        points3d:  (N, 3) numpy array 或 open3d PointCloud
        output_dir: 输出目录
        fmt:       "txt" 或 "bin"（bin 需 colmap 命令行）
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- cameras.txt ---
    cam_path = os.path.join(output_dir, "cameras.txt")
    with open(cam_path, "w") as f:
        f.write("# Camera list\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\n")
        for cam in cameras:
            params_str = " ".join(f"{v}" for v in cam["params"])
            f.write(f"{cam['camera_id']} {cam['model']} {cam['width']} {cam['height']} {params_str}\n")

    # --- images.txt ---
    img_path = os.path.join(output_dir, "images.txt")
    with open(img_path, "w") as f:
        f.write("# Image list\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n\n")
        for img in images:
            f.write(
                f"{img['image_id']} "
                f"{img['qw']} {img['qx']} {img['qy']} {img['qz']} "
                f"{img['tx']} {img['ty']} {img['tz']} "
                f"{img['camera_id']} {img['name']}\n\n"
            )

    # --- points3D.txt ---
    pts_path = os.path.join(output_dir, "points3D.txt")
    if hasattr(points3d, "points"):
        pts = np.asarray(points3d.points)
    else:
        pts = np.asarray(points3d)
    with open(pts_path, "w") as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n\n")
        for i, p in enumerate(pts):
            f.write(f"{i+1} {p[0]} {p[1]} {p[2]} 128 128 128 1.0\n")

    print(f"[INFO] TXT model exported to: {output_dir}")

    # --- TXT → BIN 转换 ---
    if fmt == "bin":
        cmd = [
            "colmap", "model_converter",
            "--input_path", output_dir,
            "--output_path", output_dir,
            "--output_type", "BIN",
        ]
        print("[INFO] Converting TXT → BIN ...")
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print(f"[INFO] BIN model exported to: {output_dir}")


# ==============================================================================
#  5. 可视化（调试用）
# ==============================================================================

def _create_camera_frustum(pose, intrinsics, scale=0.2):
    """根据位姿和内参构建相机视锥 LineSet"""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    w, h = intrinsics["width"], intrinsics["height"]

    z = -scale
    corners = np.array([
        [(0 - cx) / fx * z, (0 - cy) / fy * z, z],
        [(w - cx) / fx * z, (0 - cy) / fy * z, z],
        [(w - cx) / fx * z, (h - cy) / fy * z, z],
        [(0 - cx) / fx * z, (h - cy) / fy * z, z],
    ])
    origin = np.zeros((1, 3))
    points = np.vstack((origin, corners))

    R = quat_to_rot(pose["q"])
    t = pose["t"]
    points = (R @ points.T).T + t

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    colors = [[1, 0, 0] for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize(pcd, poses, intrinsics):
    """在 open3d 窗口中可视化点云与相机位姿"""
    geometries = [pcd]
    for pose in poses:
        geometries.append(_create_camera_frustum(pose, intrinsics))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 2.0

    vis.run()
    vis.destroy_window()



# ==============================================================================
#  入口
# ==============================================================================

if __name__ == "__main__":
    parser = ArgumentParser("convert_ar_to_colmap")
    parser.add_argument("--input", type=str, required=True,
                        help="AR output dir containing pose, points and intrinsic files")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--camera", default="PINHOLE", type=str)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vertical", action="store_true",
                        help="手机竖屏拍摄（横竖屏差90°）")
    parser.add_argument("--align-ground", action="store_true",
                        help="拟合地平面并将场景旋转至 Z 轴朝上")
    parser.add_argument("--da3-output", type=str, default=None,
                        help="DA3 输出的 COLMAP 模型目录（含 cameras.txt），默认同 --output")
    args = parser.parse_args()

    # 1. 读取数据
    pcd, poses, intrinsics = read_ar_data(args.input)

    # 2. 地面对齐（可选）
    if args.align_ground:
        cam_positions = np.array([pose["t"] for pose in poses])
        cloud_points = np.asarray(pcd.points)
        normal, centroid, offset = fit_ground_plane(cam_positions, cloud_points)
        pcd, poses, R_align = align_scene_to_plane(pcd, poses, normal)

    # 3. 转换为 COLMAP 格式
    cameras, images = convert_poses_to_colmap(
        poses, intrinsics,
        vertical=args.vertical,
        camera_model=args.camera,
    )

    # 4. 写出 COLMAP 模型
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.input, output_path)

    write_colmap_model(cameras, images, pcd, output_path, fmt="txt")

    # 5. 重命名关键帧目录（如果存在）
    keyframes_dir = os.path.join(args.input, "FrameExtraction")
    if os.path.exists(keyframes_dir):
        os.rename(keyframes_dir, os.path.join(args.input, "input"))

    # 6. 可视化（可选）
    if args.vis:
        visualize(pcd, poses, intrinsics)

    # 7. 根据实际图像尺寸更新 cameras.txt 相机内参
    da3_out = args.da3_output if args.da3_output else output_path
    