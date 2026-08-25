import os
import numpy as np
from pose_utils import quat_to_rot
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

    pts_cam_filtered = pts_cam[valid]

    if len(pts_cam_filtered) == 0:
        return np.zeros(len(voxels), dtype=bool)

    w, h = camera.width, camera.height
    if camera.model == "SIMPLE_RADIAL" or camera.model == "SIMPLE_PINHOLE":
        # COLMAP 的 SIMPLE_RADIAL 模型没有畸变参数，直接按针孔投影即可
        fx = camera.params[0]
        fy = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
    elif camera.model == "PINHOLE" or camera.model == "RADIAL" or camera.model == "OPENCV" or camera.model == "OPENCV_FISHEYE":
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


    u = fx * pts_cam_filtered[:, 0] / pts_cam_filtered[:, 2] + cx
    v = fy * pts_cam_filtered[:, 1] / pts_cam_filtered[:, 2] + cy

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    vis = np.zeros(len(voxels), dtype=bool)
    # Map filtered indices back to original voxels indices
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
    """计算两个可见性向量的 Jaccard 相似度（直接计算）"""
    inter = np.logical_and(vis1, vis2).sum()
    union = np.logical_or(vis1, vis2).sum()
    return inter / union if union > 0 else 0.0

def compute_all_overlaps(vis_cache, pose_items):
    """
    预计算所有图像对的 overlap，使用上三角矩阵避免重复计算
    返回: {(pose_i_name, pose_j_name): overlap_value, ...}
    """
    overlap_cache = {}
    n = len(pose_items)
    
    for idx_i in range(n):
        pose_i_name = pose_items[idx_i][1].name
        vis_i = vis_cache[pose_i_name]
        
        for idx_j in range(idx_i + 1, n):
            pose_j_name = pose_items[idx_j][1].name
            vis_j = vis_cache[pose_j_name]
            
            # 计算 overlap，存储到缓存
            inter = np.logical_and(vis_i, vis_j).sum()
            union = np.logical_or(vis_i, vis_j).sum()
            overlap = inter / union if union > 0 else 0.0
            overlap_cache[(pose_i_name, pose_j_name)] = overlap
    
    return overlap_cache

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
    max_angle=np.radians(70),
    min_overlap=0.1,
    camera_info_cache=None,
    overlap_cache=None,
    overlap_value=None
):
    """
    camera_info_cache: dict[name] → (C, v) 相机中心和方向（可选，用于加速）
    overlap_cache: dict[(name1, name2)] → overlap_value（可选，预计算的 overlap）
    overlap_value: 直接传入的 overlap 值（可选，当已预计算时使用）
    """

    # -------------------------
    # 相机中心 & 方向
    # -------------------------
    if camera_info_cache is not None:
        C1, v1 = camera_info_cache[pose1.name]
        C2, v2 = camera_info_cache[pose2.name]
    else:
        C1, v1 = get_camera_center_and_dir(pose1)
        C2, v2 = get_camera_center_and_dir(pose2)

    # -------------------------
    # angle
    # -------------------------
    cos_theta = np.clip(np.dot(v1, v2), -1, 1)
    angle = np.arccos(cos_theta)
    # print(f"Angle between {pose1.name} and {pose2.name}: {np.degrees(angle):.2f} degrees")
    if angle > max_angle:
        return False

    # -------------------------
    # baseline
    # -------------------------
    # baseline = np.linalg.norm(C1 - C2)
    # if baseline < min_baseline or baseline > max_baseline:
    #     return False
    # -------------------------
    # 视锥 overlap（使用缓存或直接值）
    # -------------------------
    if overlap_value is not None:
        overlap = overlap_value
    elif overlap_cache is not None:
        key = (pose1.name, pose2.name)
        overlap = overlap_cache.get(key, 0.0)
    else:
        return False  # 没有 overlap 信息时无法判断

    if overlap < min_overlap:
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

def compute_adaptive_voxel_size(poses, min_size=0.05, max_size=1.0):
    """
    自适应计算 voxel size。

    基本逻辑：对每个相机，计算其相机中心到最近邻相机中心的距离，
    然后对所有相机的最近邻距离取均值，作为 voxel size。
    返回的 voxel size 会被限制在 [min_size, max_size] 范围内。

    参数：
        poses: dict[int, Image]    COLMAP 格式的相机位姿（含 qvec / tvec）
        min_size: float            voxel size 下限
        max_size: float            voxel size 上限

    返回：
        voxel_size: float          自适应的 voxel size
    """
    centers = np.array([
        get_camera_center_and_dir(pose)[0] for _, pose in poses.items()
    ])

    n = len(centers)
    if n < 2:
        return min_size

    nearest_dists = []
    for i in range(n):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists = np.delete(dists, i)  # 排除自身
        nearest_dists.append(dists.min())

    voxel_size = float(np.mean(nearest_dists))

    return float(np.clip(voxel_size, min_size, max_size))


def compute_matched_image_pairs_by_pose_prior(pose_prior_path, output_txt, voxel_size=None, max_angle=70, min_overlap=0.1):

    # 读取 COLMAP 数据
    cameras, poses, points3D = read_model(pose_prior_path)

    # 未显式指定 voxel size 时，基于相机间距自适应计算
    if voxel_size is None:
        voxel_size = compute_adaptive_voxel_size(poses)
        print(f"[INFO] Adaptive voxel size computed: {voxel_size:.4f}")

    # 构建体素
    points3D_xyz = np.array([point.xyz for point in points3D.values()])

    # 构建体素前，将点云随机下采样至 10w 个点，避免点数过多导致体素计算开销过大
    max_points = 10
    if len(points3D_xyz) > max_points:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(points3D_xyz), max_points, replace=False)
        points3D_xyz = points3D_xyz[sample_idx]

    _, voxels = build_voxel_grid(points3D_xyz, voxel_size)

    vis_cache = precompute_visibility(poses, cameras, voxels)

    # 预计算所有相机的中心和方向（加速匹配）
    camera_info_cache = {}
    for _, pose in poses.items():
        camera_info_cache[pose.name] = get_camera_center_and_dir(pose)

    # 预计算所有对的 overlap 值（关键优化）
    pose_items = list(poses.items())
    overlap_cache = compute_all_overlaps(vis_cache, pose_items)
    print(f"[INFO] Precomputed {len(overlap_cache)} overlap values")

    # 构建匹配图
    pairs = []
    images = []  # 每个元素为 (image_id, image_path, camera_id)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w") as f:
        # 只遍历上三角，使用预计算的 overlap
        for idx_i in range(len(pose_items)):
            img_id_i, pose_i = pose_items[idx_i]
            images.append((img_id_i, pose_i.name, pose_i.camera_id))

            for idx_j in range(idx_i + 1, len(pose_items)):
                img_id_j, pose_j = pose_items[idx_j]
                # 从缓存中取 overlap 值
                overlap_value = overlap_cache.get((pose_i.name, pose_j.name), 0.0)

                if is_matchable_advanced(
                    pose_i, pose_j, 
                    max_angle=np.radians(max_angle), 
                    min_overlap=min_overlap, 
                    camera_info_cache=camera_info_cache,
                    overlap_value=overlap_value
                ):
                    pairs.append((pose_i.name, pose_j.name))
                    f.write(f"{pose_i.name} {pose_j.name}\n")
    print(f"[INFO] Matched image pairs computed and saved to {output_txt}, total pairs: {len(pairs)}")

    return cameras, images