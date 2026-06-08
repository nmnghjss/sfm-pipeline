import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import os
import subprocess

# -----------------------------
# 读取点云 PLY
# -----------------------------
def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[INFO] Loaded point cloud: {len(pcd.points)} points")
    pcd.paint_uniform_color([0, 1, 0])  # 绿色
    return pcd


# -----------------------------
# 读取位姿 PLY（自定义解析）
# -----------------------------
def load_poses(ply_path):
    poses = []

    with open(ply_path, "r") as f:
        lines = f.readlines()

    # 找 header 结束
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
            "q": np.array([qw, qx, qy, qz]),  # 注意顺序：w, x, y, z
            "name": name
        })

    print(f"[INFO] Loaded poses: {len(poses)} cameras")
    return poses


# -----------------------------
# 读取内参
# -----------------------------
def load_intrinsics(txt_path):
    params = {}

    with open(txt_path, "r") as f:
        for line in f:
            k, v = line.strip().split("=")
            params[k] = float(v)

    return params


# -----------------------------
# 四元数 -> 旋转矩阵
# -----------------------------
def quat_to_rot(q):
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ])
    return R


def rot_to_quat(R):
    """
    Convert rotation matrix to quaternion (qw, qx, qy, qz)
    输入:
        R: 3x3 rotation matrix
    输出:
        qw, qx, qy, qz
    """
    R = np.asarray(R, dtype=np.float64)
    q = np.zeros(4, dtype=np.float64)

    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        # 找主对角最大元素
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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

    # 归一化（很重要）
    q /= np.linalg.norm(q)

    return q[0], q[1], q[2], q[3]


# -----------------------------
# 构建相机 frustum
# -----------------------------
def create_camera_frustum(pose, intrinsics, scale=0.2):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    w = intrinsics["width"]
    h = intrinsics["height"]

    # 相机坐标系下的4个角点
    z = -scale
    corners = np.array([
        [(0 - cx) / fx * z, (0 - cy) / fy * z, z],
        [(w - cx) / fx * z, (0 - cy) / fy * z, z],
        [(w - cx) / fx * z, (h - cy) / fy * z, z],
        [(0 - cx) / fx * z, (h - cy) / fy * z, z],
    ])

    origin = np.zeros((1, 3))
    points = np.vstack((origin, corners))

    # 旋转和平移
    R = quat_to_rot(pose["q"])
    t = pose["t"]

    points = (R @ points.T).T + t

    # 线框连接
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    colors = [[1, 0, 0] for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


# -----------------------------
# 主函数
# -----------------------------
def visualize(pcd, poses, intrinsics):

    geometries = [pcd]

    for pose in poses:
        cam = create_camera_frustum(pose, intrinsics)
        geometries.append(cam)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加几何体
    for g in geometries:
        vis.add_geometry(g)

    # 设置渲染参数
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # 黑背景
    opt.point_size = 2.0 

    # 运行
    vis.run()
    vis.destroy_window()

def export_to_colmap(pcd, poses, intrinsics, output_dir, fmt="txt", camera_model="SIMPLE_RADIAL", vertical=False):
    """
    导出到COLMAP格式
    c2w→w2c + 必要的180°相机前向修正
    portrait: 手机是否竖屏拍摄（横竖屏差90°，需额外旋转）
    """
    """
    fmt: "txt" or "bin"
    """

    os.makedirs(output_dir, exist_ok=True)
    # -------------------------
    # cameras.txt
    # -------------------------
    cam_path = os.path.join(output_dir, "cameras.txt")

    if intrinsics is not None:
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        # cx = intrinsics["cx"]
        # cy = intrinsics["cy"]
        w = int(intrinsics["width"])
        h = int(intrinsics["height"])
        cx = w / 2
        cy = h / 2

        with open(cam_path, "w") as f:
            f.write("# Camera list\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\n")
            if camera_model == "PINHOLE":
                f.write(f"1 {camera_model} {w} {h} {fx} {fy} {cx} {cy}\n")
            elif camera_model == "SIMPLE_PINHOLE":
                f.write(f"1 {camera_model} {w} {h} {max(fx, fy)} {cx} {cy}\n")
            elif camera_model == "RADIAL":
                f.write(f"1 {camera_model} {w} {h} {max(fx, fy)} {cx} {cy} 0.0000 0.0000\n")
            elif camera_model == "SIMPLE_RADIAL":
                f.write(f"1 {camera_model} {w} {h} {max(fx, fy)} {cx} {cy} 0.0000\n")
            else:
                f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # -------------------------
    # images.txt
    # -------------------------
    img_path = os.path.join(output_dir, "images.txt")

    with open(img_path, "w") as f:
        f.write("# Image list\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n\n")

        FLIP_YZ = np.diag([1, -1, -1])

        for i, pose in enumerate(poses):
            R = quat_to_rot(pose["q"])
            t = pose["t"]

            # 手机横竖屏差90°（仅竖屏时执行）
            if vertical:
                theta = np.radians(90)
                c, s = np.cos(theta), np.sin(theta)
                R_z90 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                R = R @ R_z90  # 绕相机Z轴旋转90°（修正横竖屏）
            # c2w → w2c 转换
            R_w2c = R.T
            t_w2c = -R_w2c @ t
            # 绕X轴180°翻转（修正AR相机前向-Z → COLMAP前向+Z）
            R_fix = np.diag([1.0, -1.0, -1.0])
            R_w2c = R_fix @ R_w2c
            t_w2c = R_fix @ t_w2c
                
            qw, qx, qy, qz = rot_to_quat(R_w2c)

            name = pose["name"]

            f.write(
                f"{i+1} {qw} {qx} {qy} {qz} "
                f"{t_w2c[0]} {t_w2c[1]} {t_w2c[2]} "
                f"1 {name}\n\n"
            )

    # -------------------------
    # points3D.txt
    # -------------------------
    pts_path = os.path.join(output_dir, "points3D.txt")

    pts = np.asarray(pcd.points)

    with open(pts_path, "w") as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n\n")

        for i, p in enumerate(pts):
            f.write(
                f"{i+1} {p[0]} {p[1]} {p[2]} 128 128 128 1.0\n"
            )

    print(f"[INFO] TXT model exported to: {output_dir}")

    # -------------------------
    if fmt == "bin":
        cmd = [
            "colmap", "model_converter",
            "--input_path", output_dir,
            "--output_path", output_dir,
            "--output_type", "BIN"
        ]

        print("[INFO] Converting TXT → BIN ...")
        # subprocess.run(cmd, check=True)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            shell=False  # Windows, using list command
        )
        print(f"[INFO] BIN model exported to: {output_dir}")

# -----------------------------
# 入口
# -----------------------------
if __name__ == "__main__":

    argpaser = ArgumentParser("visualize ar output")
    argpaser.add_argument("--input", type=str, required=True, help="ar output dir, must containing pose, points and intrinsic files")
    argpaser.add_argument("--output", type=str, required=True)
    argpaser.add_argument("--camera", default="SIMPLE_RADIAL", type=str)
    argpaser.add_argument("--vis", action="store_true")
    argpaser.add_argument("--vertical", action="store_true", help="手机竖屏拍摄（横竖屏差90°）")

    args = argpaser.parse_args()

    points_path = os.path.join(args.input, "scan_all_pointcloud.ply")
    poses_path = os.path.join(args.input, "scan_all_pose.ply")
    intrinsic_path = os.path.join(args.input, "scan_camera_intrinsics.txt")

    pcd = load_point_cloud(points_path)
    poses = load_poses(poses_path)

    if os.path.exists(intrinsic_path):
        intrinsics = load_intrinsics(intrinsic_path)
    else:
        intrinsics = None

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.input, output_path)

    os.makedirs(output_path, exist_ok=True)

    export_to_colmap(pcd, poses, intrinsics, output_path, fmt="txt", vertical=args.vertical)

    keyframesDir = os.path.join(args.input, "FrameExtraction")
    if os.path.exists(keyframesDir):
        newDir = os.path.join(args.input, "input")
        os.rename(keyframesDir, newDir)

    if args.vis:
        visualize(pcd, poses, intrinsics)