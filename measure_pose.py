import os
import sys
import numpy as np
from argparse import ArgumentParser

from read_write_model import qvec2rotmat, read_model, rotmat2qvec, Image as ColmapImage
import matplotlib.pyplot as plt


class PoseError:
    def __init__(
        self,
        ate_error_rmse=0.0,
        ate_error_mean=0.0,
        ate_error_median=0.0,
        ate_error_std=0.0,
        ate_error_max=0.0,
        ate_error_p90=0.0,
        rotate_angle_error_rmse=0,
        rotate_angle_error_mean=0.0,
        rotate_angle_error_median=0.0,
        rotate_angle_error_std=0.0,
        rotate_angle_error_max=0.0,
        rotate_angle_error_p90=0.0,
        registered_diff=0,
        registered_ratio_diff=0.0,
        base_registered_num=0,
        update_registered_num=0,
        base_registered_ratio=0.0,
        update_registered_ratio=0.0,
    ):
        self.ate_error_rmse = ate_error_rmse
        self.ate_error_mean = ate_error_mean
        self.ate_error_median = ate_error_median
        self.ate_error_std = ate_error_std
        self.ate_error_max = ate_error_max
        self.ate_error_p90 = ate_error_p90
        
        self.rotate_angle_error_rmse = rotate_angle_error_rmse
        self.rotate_angle_error_mean = rotate_angle_error_mean
        self.rotate_angle_error_median = rotate_angle_error_median
        self.rotate_angle_error_std = rotate_angle_error_std
        self.rotate_angle_error_max = rotate_angle_error_max
        self.rotate_angle_error_p90 = rotate_angle_error_p90

        self.registered_diff = registered_diff
        self.registered_ratio_diff = registered_ratio_diff

        self.base_registered_num = base_registered_num
        self.update_registered_num = update_registered_num
        self.base_registered_ratio = base_registered_ratio
        self.update_registered_ratio = update_registered_ratio



def umeyama_align(X, Y, with_scale=True):
    # X, Y: Nx3 arrays; find s, R, t so that Y ~ s*R*X + t
    assert X.shape == Y.shape
    n, m = X.shape
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    Sxx = (Xc.T @ Xc) / n
    cov = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        varX = np.sum(Xc ** 2) / n
        s = np.trace(np.diag(D) @ S) / varX
    else:
        s = 1.0

    t = muY - s * R @ muX
    return s, R, t


def umeyama_align_ransac(X_dict, Y_dict, with_scale=True, max_iters=1000, inlier_threshold=1.0, min_inliers=None, random_seed=None):
    """
    使用RANSAC对umeyama对齐进行鲁棒估计，处理外点。

    Args:
        X_dict, Y_dict: 字典，键为相同标识符（如图像名称），值为对应的 3D 点坐标 (length-3 array)
        with_scale: 是否估计缩放
        max_iters: RANSAC最大迭代次数
        inlier_threshold: 判断内点的距离阈值（米）
        min_inliers: 最少内点数量（默认为 max(3, 50% * N)）
        random_seed: 随机种子，便于复现

    Returns:
        s, R, t, inlier_mask_dict
        - s: 缩放因子
        - R: 3x3 旋转矩阵
        - t: 平移向量 (length-3)
        - inlier_mask_dict: 字典，键与输入相同，值为布尔值标记内点
    """
    # 转换为数组便于计算
    keys = list(X_dict.keys())
    X = np.array([X_dict[k] for k in keys])
    Y = np.array([Y_dict[k] for k in keys])
    
    assert X.shape == Y.shape
    print("X.shape: ", X.shape)
    n, m = X.shape

    # 如果点少于3个，无法可靠估计，返回默认恒等变换并打印警告
    if n < 3:
        print("警告: 输入点数少于3，无法进行RANSAC估计，返回 s=1, R=I, t=0")
        s = 1.0
        R = np.eye(3)
        t = np.zeros(3)
        inlier_mask_dict = {k: True for k in keys}
        return s, R, t, inlier_mask_dict

    if min_inliers is None:
        min_inliers = max(3, int(0.5 * n))

    rng = np.random.RandomState(random_seed)
    best_inliers = None
    best_model = None
    best_count = 0

    for _ in range(max_iters):
        # 从字典键中随机选择采样
        try:
            sample_keys = rng.choice(keys, size=min_inliers, replace=False)
        except ValueError:
            continue

        # 获取采样的点
        Xs = np.array([X_dict[k] for k in sample_keys])
        Ys = np.array([Y_dict[k] for k in sample_keys])

        # 避免退化样本（共面/共线等）: 检查中心化后的秩
        if np.linalg.matrix_rank(Xs - Xs.mean(axis=0)) < 3:
            continue

        # 估计模型
        try:
            s_try, R_try, t_try = umeyama_align(Xs, Ys, with_scale=with_scale)
        except Exception:
            continue

        # 将所有 X 变换到估计的模型下并计算与 Y 的距离
        X_trans = (s_try * (R_try @ X.T)).T + t_try
        dists = np.linalg.norm(X_trans - Y, axis=1)
        # 计算并打印dists的统计信息
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        print(f"图像坐标 对齐后，距离误差统计信息: 均值={mean_dist:.3f} 米, 标准差={std_dist:.3f} 米")
        inliers = dists <= inlier_threshold
        count = int(inliers.sum())
        
        # 统计内点的误差均值和标准差
        inlier_dists = dists[inliers]
        if len(inlier_dists) > 0:
            inlier_mean = np.mean(inlier_dists)
            inlier_std = np.std(inlier_dists)
            print(f"内点数量：{count}，内点距离误差统计信息: 均值={inlier_mean:.3f} 米, 标准差={inlier_std:.3f} 米")

        # 更新最优模型
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (s_try, R_try, t_try)

            # 如果达到所有点都是内点，提前退出
            if best_count >= n:
                break

    # 如果没有找到足够的内点，则回退到对全部点的估计
    if best_inliers is None or best_count < min_inliers:
        s, R, t = umeyama_align(X, Y, with_scale=with_scale)
        inlier_mask_dict = {k: True for k in keys}
        return s, R, t, inlier_mask_dict

    # 打印最佳内点数量、误差均值和标准差
    if best_inliers is not None:
        best_inlier_dists = np.linalg.norm((best_model[0] * (best_model[1] @ X.T)).T + best_model[2] - Y, axis=1)[best_inliers]
        best_inlier_mean = np.mean(best_inlier_dists)
        best_inlier_std = np.std(best_inlier_dists)
        print(f"最佳内点数量: {best_count}/{n}, 误差均值: {best_inlier_mean:.3f} 米, 标准差: {best_inlier_std:.3f} 米")

    # 使用所有内点重新拟合以获得更好估计
    inlier_keys = [k for k, inlier in zip(keys, best_inliers) if inlier]
    X_inliers = np.array([X_dict[k] for k in inlier_keys])
    Y_inliers = np.array([Y_dict[k] for k in inlier_keys])
    s_ref, R_ref, t_ref = umeyama_align(X_inliers, Y_inliers, with_scale=with_scale)
    
    # 转换为字典格式返回内点掩码
    inlier_mask_dict = {k: inlier for k, inlier in zip(keys, best_inliers)}
    return s_ref, R_ref, t_ref, inlier_mask_dict


def align_camera_pose(Rwc, twc, R_align, scale, shift):
    C = -Rwc.T @ twc
    C_new = scale * (R_align @ C) + shift

    # 新旋转（世界坐标变换 ⇒ 右乘 R^T）
    R_new = Rwc @ R_align.T
    qvec_new = rotmat2qvec(R_new)
    t_new = -R_new @ C_new

    return R_new, t_new, qvec_new    


def compute_alignment_error(base_images_pose:dict, update_images_pose:dict, visualize=False):
    errors_ate = []
    rotate_angle_errors = []
    for name, base_image in base_images_pose.items():
        if name not in update_images_pose:
            continue
        update_image = update_images_pose[name]

        Rwc_base = qvec2rotmat(base_image.qvec)
        twc_base = base_image.tvec
        C_base = -Rwc_base.T @ twc_base  # 相机中心

        Rwc_update = qvec2rotmat(update_image.qvec)
        twc_update = update_image.tvec
        C_update = -Rwc_update.T @ twc_update  # 相机中心

        # 计算对齐后的相机中心位置误差
        error = np.linalg.norm(C_base - C_update)
        errors_ate.append(error)

        # 计算对齐后的旋转误差（度）
        R_diff = Rwc_base @ Rwc_update.T 
        angle_error_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        angle_error_deg = np.degrees(angle_error_rad)
        rotate_angle_errors.append(angle_error_deg)

    if len(errors_ate) > 0:
        ate_error_mean = np.mean(errors_ate)
        ate_error_std = np.std(errors_ate)
        ate_error_rmse = np.sqrt(np.mean(np.array(errors_ate) ** 2))
        ate_error_median = np.median(errors_ate)
        ate_error_max = np.max(errors_ate)  
        ate_error_p90 = np.percentile(errors_ate, 90)
        angle_error_mean = np.mean(rotate_angle_errors)
        angle_error_std = np.std(rotate_angle_errors)
        angle_error_median = np.median(rotate_angle_errors)
        angle_error_max = np.max(rotate_angle_errors)
        angle_error_rmse = np.sqrt(np.mean(np.array(rotate_angle_errors) ** 2))
        angle_error_p90 = np.percentile(rotate_angle_errors, 90)

        print(f"相机中心位置误差统计信息: 均值={ate_error_mean:.3f} 米, 标准差={ate_error_std:.3f} 米, rmse = {ate_error_rmse:.3f} 米, 90%分位数 = {ate_error_p90:.3f} 米，最大误差 = {ate_error_max:.3f} 米")
        print(f"相机旋转误差统计信息: 均值={angle_error_mean:.3f} 度, 标准差={angle_error_std:.3f} 度, rmse = {angle_error_rmse:.3f} 度, median = {angle_error_median:.3f} 度， 90%分位数 = {angle_error_p90}, 最大误差 = {angle_error_max}")
        
        pose_error = PoseError()
        pose_error.ate_error_mean = ate_error_mean
        pose_error.ate_error_std = ate_error_std
        pose_error.ate_error_rmse = ate_error_rmse
        pose_error.ate_error_median = ate_error_median
        pose_error.ate_error_max = ate_error_max
        pose_error.ate_error_p90 = ate_error_p90

        pose_error.rotate_angle_error_mean = angle_error_mean
        pose_error.rotate_angle_error_std = angle_error_std
        pose_error.rotate_angle_error_rmse = angle_error_rmse
        pose_error.rotate_angle_error_median = angle_error_median
        pose_error.rotate_angle_error_max = angle_error_max
        pose_error.rotate_angle_error_p90 = angle_error_p90

        if visualize:
            # 位置误差绘图
            plt.figure(figsize=(12, 6))
            plt.plot(errors_ate, label='Position Error', marker='o', markersize=3, linewidth=1.5)
            plt.axhline(y=ate_error_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {ate_error_mean:.3f} m')
            plt.fill_between(range(len(errors_ate)), ate_error_mean - 3*ate_error_std, ate_error_mean + 3*ate_error_std, 
                            alpha=0.2, color='r', label=f'±3xstd: {ate_error_std*3:.3f} m')
            plt.xlabel('Image Index', fontsize=12)
            plt.ylabel('Position Error (meters)', fontsize=12)
            plt.title('Camera Position Error (ATE)', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 旋转误差绘图
            plt.figure(figsize=(12, 6))
            plt.plot(rotate_angle_errors, label='Rotation Error', marker='s', markersize=3, linewidth=1.5)
            plt.axhline(y=angle_error_mean, color='g', linestyle='--', linewidth=2, label=f'Mean: {angle_error_mean:.3f}°')
            plt.fill_between(range(len(rotate_angle_errors)), angle_error_mean - 3*angle_error_std, 
                            angle_error_mean + 3*angle_error_std, alpha=0.2, color='g', label=f'±3xstd: {angle_error_std*3:.3f}°')
            plt.xlabel('Image Index', fontsize=12)
            plt.ylabel('Rotation Error (degrees)', fontsize=12)
            plt.title('Camera Rotation Error', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return pose_error
    else:
        print("没有找到匹配的图像进行误差计算。")
        return None

def align_and_compute_error(base_images_pose:dict, update_images_pose:dict, visualize=False):

    register_img_num_diff = len(update_images_pose) - len(base_images_pose)

    base_images_name_pose = {}
    update_images_name_pose = {}

    for image_id, image in base_images_pose.items():
        # 去除文件名后缀
        name = image.name
        if '.' in name:
            name = '.'.join(name.split('.')[:-1])
        base_images_name_pose[name] = image

    for image_id, image in update_images_pose.items():
        # 去除文件名后缀
        name = image.name
        if '.' in name:
            name = '.'.join(name.split('.')[:-1])
        update_images_name_pose[name] = image

    base_images_pose = base_images_name_pose
    update_images_pose = update_images_name_pose

    base_images_pos = {}
    update_images_pos = {}
    for name, base_image in base_images_pose.items():
        if name not in update_images_pose:
            continue
        update_image = update_images_pose[name]

        Rwc_base = qvec2rotmat(base_image.qvec)
        twc_base = base_image.tvec
        C_base = -Rwc_base.T @ twc_base  # 相机中心
        base_images_pos[name] = C_base

        Rwc_update = qvec2rotmat(update_image.qvec)
        twc_update = update_image.tvec
        C_update = -Rwc_update.T @ twc_update  # 相机中心
        update_images_pos[name] = C_update

    # base_images_pos = np.array(base_images_pos)
    # update_images_pos = np.array(update_images_pos)
    if len(base_images_pos) < 3 or len(update_images_pos) < 3:
        errors = PoseError()
        errors.ate_error_max = 10000
        errors.rotate_angle_error_max = 10000
        return errors


    ransac_max_iters = 1000
    inlier_threshold = 2  # 米
    min_inliers = max(3, int(0.5 * len(base_images_pos)))  # 至少3个内点，或50%的点
    scale, R_align, shift, inliers = umeyama_align_ransac(base_images_pos, update_images_pos, max_iters=ransac_max_iters, inlier_threshold=inlier_threshold, min_inliers=min_inliers)
    print("scale:", scale)
    print("R:\n", R_align)
    print("t:", shift)        

    # 转换相机位姿并计算误差
    for name, image in base_images_pose.items():
        Rcw = qvec2rotmat(image.qvec)
        tcw = image.tvec
        Rcw_new, tcw_new, qvec_new = align_camera_pose(Rcw, tcw, R_align, scale, shift)

        base_images_pose[name] = ColmapImage(
            id=image.id,
            qvec=qvec_new,
            tvec=tcw_new,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )
    
    pose_error = compute_alignment_error(base_images_pose, update_images_pose, visualize)
    pose_error.registered_diff = register_img_num_diff
    
    return pose_error


if __name__ == "__main__":

    args = ArgumentParser(description="Measure camera poses from COLMAP output")
    args.add_argument("--colmap_output_base", "-b", type=str, required=True, help="Path to the base COLMAP output directory")
    args.add_argument("--colmap_output_update", "-u", type=str, default=None, help="Path to the upodate COLMAP output directory")
    args.add_argument("--visualize", "-vis", action='store_true', default=False, help="Whether to visualize the alignment errors")
    args = args.parse_args()

    try:
        base_cameras, base_images, base_points3D = read_model(args.colmap_output_base)
        update_cameras, update_images, update_points3D = read_model(args.colmap_output_update)

        errors = align_and_compute_error(base_images, update_images, visualize=args.visualize)
        print("Pose error:", vars(errors) if errors is not None else "No error computed.")


    except FileNotFoundError as e:
        print(str(e))


