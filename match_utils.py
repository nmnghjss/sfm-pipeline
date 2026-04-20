import os
import numpy as np
from convert_ar_to_colmap import quat_to_rot
from read_write_model import read_model

def build_voxel_grid(points, voxel_size):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    dims = ((maxs - mins) / voxel_size).astype(int) + 1

    grid = {}
    voxel_centers = []

    for p in points:
        idx = tuple(((p - mins) / voxel_size).astype(int))
        grid[idx] = True

    for idx in grid.keys():
        center = mins + (np.array(idx) + 0.5) * voxel_size
        voxel_centers.append(center)

    return grid, np.array(voxel_centers)

def compute_voxel_visibility_colmap(voxels, pose, camera):
    """
    COLMAP 坐标系版本
    """
    R = quat_to_rot(pose.qvec)
    t = pose.tvec

    # world → cam
    pts_cam = (R @ voxels.T).T + t

    z = pts_cam[:, 2]
    valid = z > 0.1

    pts_cam = pts_cam[valid]

    if len(pts_cam) == 0:
        return np.zeros(len(voxels), dtype=bool)

    w, h = camera.width, camera.height
    if camera.model == "SIMPLE_RADIAL" or camera.model == "SIMPLE_PINHOLE":
        # COLMAP 的 SIMPLE_RADIAL 模型没有畸变参数，直接按针孔投影即可
        fx = camera.params[0]
        fy = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
    elif camera.model == "PINHOLE" or camera.model == "RADIAL" or camera.model == "OPENCV":
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
    elif camera.model == "SIMPLE_PINHOLE":
        fx = camera.params[0]
        fy = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
    else:
        raise NotImplementedError(f"Unsupported camera model: {camera.model}")


    u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    vis = np.zeros(len(voxels), dtype=bool)
    vis[np.where(valid)[0][in_img]] = True

    return vis

def compute_voxel_visibility(voxels, pose, intrinsics):
    R = quat_to_rot(pose["q"])
    t = pose["t"]

    # c2w → w2c
    R_w2c = R.T
    t_w2c = -R_w2c @ t

    # AR → COLMAP
    FLIP_Z = np.diag([1, -1, -1])
    R_w2c = FLIP_Z @ R_w2c
    t_w2c = FLIP_Z @ t_w2c

    pts_cam = (R_w2c @ voxels.T).T + t_w2c

    z = pts_cam[:, 2]
    mask = z > 0.1

    pts_cam = pts_cam[mask]

    if len(pts_cam) == 0:
        return np.zeros(len(voxels), dtype=bool)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    w = intrinsics["width"]
    h = intrinsics["height"]

    u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    vis_mask = np.zeros(len(voxels), dtype=bool)
    vis_mask[np.where(mask)[0][in_img]] = True

    return vis_mask

def compute_voxel_overlap(vis1, vis2):
    inter = np.logical_and(vis1, vis2).sum()
    union = np.logical_or(vis1, vis2).sum()
    return inter / union if union > 0 else 0.0

def get_camera_center_and_dir(pose):
    R = quat_to_rot(pose.qvec)
    t = pose.tvec

    # camera center
    C = -R.T @ t

    # COLMAP: camera looks along +Z
    forward_cam = np.array([0, 0, 1])
    view_dir = R.T @ forward_cam

    view_dir /= np.linalg.norm(view_dir)

    return C, view_dir

def compute_view_direction(pose):
    """
    返回相机在世界坐标系下的 forward 向量
    """
    R = quat_to_rot(pose["q"])  # c2w
    forward_cam = np.array([0, 0, -1])  # AR系

    view_dir = R @ forward_cam
    view_dir /= np.linalg.norm(view_dir)

    return view_dir

def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # radians

def dot_between(v1, v2):
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)
    return dot

def build_frustum_overlap_voxel(
    pcd,
    poses,
    intrinsics,
    output_txt,
    voxel_size=0.2,
    max_pairs_per_image=20,
    overlap_thresh=0.02,
    max_angle=np.radians(45)
):
    points = np.asarray(pcd.points)

    # 构建 voxel grid
    _, voxel_centers = build_voxel_grid(points, voxel_size)

    # 每个相机的 voxel 可见性
    visibility = []
    view_dirs = []

    for pose in poses:
        vis = compute_voxel_visibility(voxel_centers, pose, intrinsics)
        visibility.append(vis)
        vdir = compute_view_direction(pose)
        view_dirs.append(vdir)        

    # 构建匹配图
    min_dot = np.cos(max_angle)
    pairs_num = 0
    with open(output_txt, "w") as f:
        for i in range(len(poses)):
            scores = []

            for j in range(i+1, len(poses)):
                if i == j:
                    continue

                # -------------------------
                # 方向约束
                # -------------------------
                dot = dot_between(view_dirs[i], view_dirs[j])
                if dot < min_dot:
                    continue

                score = compute_voxel_overlap(visibility[i], visibility[j])

                if score > overlap_thresh:
                    scores.append((j, score))

            scores.sort(key=lambda x: -x[1])
            scores = scores[:max_pairs_per_image]

            for j, _ in scores:
                f.write(f"{poses[i]['name']} {poses[j]['name']}\n")
                pairs_num += 1

    print(f"[INFO] Voxel-based frustum overlap done, initial pairs saved to {output_txt}, total pairs: {pairs_num}")

def is_matchable_advanced(
    pose1,
    pose2,
    vis_cache,
    angle_thresh_deg=45,
    overlap_thresh=0.02,
    min_baseline=0.05,
    max_baseline=10.0
):
    """
    vis_cache: dict[name] → visibility mask（提前算好）
    """

    # -------------------------
    # 相机中心 & 方向
    # -------------------------
    C1, v1 = get_camera_center_and_dir(pose1)
    C2, v2 = get_camera_center_and_dir(pose2)

    # -------------------------
    # angle
    # -------------------------
    cos_theta = np.clip(np.dot(v1, v2), -1, 1)
    angle = np.degrees(np.arccos(cos_theta))
    if angle > angle_thresh_deg:
        return False

    # -------------------------
    # baseline
    # -------------------------
    # baseline = np.linalg.norm(C1 - C2)
    # if baseline < min_baseline or baseline > max_baseline:
    #     return False

    # -------------------------
    # 朝向一致性
    # -------------------------
    # dir12 = (C2 - C1)
    # dir12 /= np.linalg.norm(dir12)

    # if np.dot(v1, dir12) < 0:
    #     return False

    # if np.dot(v2, -dir12) < 0:
    #     return False

    # -------------------------
    # 视锥 overlap（核心）
    # -------------------------
    vis1 = vis_cache[pose1.name]
    vis2 = vis_cache[pose2.name]

    overlap = compute_voxel_overlap(vis1, vis2)

    if overlap < overlap_thresh:
        return False

    return True

def precompute_visibility(poses, cameras, voxels):
    vis_cache = {}

    for _, pose in poses.items():
        cam = cameras[pose.camera_id]
        vis = compute_voxel_visibility_colmap(voxels, pose, cam)
        vis_cache[pose.name] = vis

    print("[INFO] Visibility precomputed")
    return vis_cache

def load_image_pairs(pairs_txt, unique=True):
    """
    从 pairs.txt 中读取图像对

    参数：
        pairs_txt: str
        unique: 是否去重（无向对）

    返回：
        pairs: List[Tuple[str, str]]
    """
    pairs = []
    seen = set()

    with open(pairs_txt, "r") as f:
        for line_id, line in enumerate(f):
            line = line.strip()

            # 跳过空行 / 注释
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] Invalid line {line_id}: {line}")
                continue

            name1, name2 = parts[0], parts[1]

            if unique:
                # 无向去重（A-B 和 B-A 视为同一对）
                key = tuple(sorted((name1, name2)))
                if key in seen:
                    continue
                seen.add(key)

            pairs.append((name1, name2))

    print(f"[INFO] Loaded {len(pairs)} image pairs")
    return pairs

def compute_matched_image_pairs_by_pose_prior(pose_prior_path, output_txt, voxel_size=0.2, overlap_thresh=0.4):

    # 读取 COLMAP 数据
    cameras, poses, points3D = read_model(pose_prior_path)

    # 构建体素
    points3D_xyz = np.array([point.xyz for point in points3D.values()])
    _, voxels = build_voxel_grid(points3D_xyz, voxel_size)

    vis_cache = precompute_visibility(poses, cameras, voxels)

    # 构建匹配图
    pairs = []
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w") as f:
        for i in range(len(poses)):
            for j in range(i+1, len(poses)):
                pose_i = poses[i+1]
                pose_j = poses[j+1]
                if is_matchable_advanced(pose_i, pose_j, vis_cache, overlap_thresh=overlap_thresh):
                    pairs.append((pose_i.name, pose_j.name))
                    f.write(f"{pose_i.name} {pose_j.name}\n")
    print(f"[INFO] Matched image pairs computed and saved to {output_txt}, total pairs: {len(pairs)}")