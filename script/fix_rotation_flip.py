"""
fix_rotation_flip.py

修复 COLMAP update 模型相对 base 模型的 ~180° 旋转误差，并写回 update 模型。

背景
----
对 base / update 两组 COLMAP 位姿做"基于相机中心"的 umeyama 对齐后：
    * 位置误差通常很小（说明相机中心已经对得很准）
    * 但旋转误差可能稳定在 ~180°（坐标轴/朝向发生翻转）

本脚本流程：
    1. 读取 base / update 两个 COLMAP sparse 模型；
    2. 用 RANSAC umeyama 做位置对齐（base -> update 坐标系），得到 scale, R_align, shift；
    3. 逐图计算残差旋转 R_diff = Rwc_base_aligned @ Rwc_update.T；
    4. 若旋转误差中位数 ~180°，则估计全局翻转旋转 R_flip（四元数平均）；
    5. 纠正 update 相机位姿（保持相机中心不变）：
            Rwc' = R_flip @ Rwc
            twc' = R_flip @ twc
    6. 写回 colmap_output_update（写前自动备份原模型）。

用法
----
    python fix_rotation_flip.py -b <base_sparse_dir> -u <update_sparse_dir> [--visualize]
"""

import os
import sys
import shutil
from datetime import datetime

import numpy as np
from argparse import ArgumentParser

from read_write_model import (
    qvec2rotmat,
    rotmat2qvec,
    read_model,
    write_model,
    detect_model_format,
    Image as ColmapImage,
)
from measure_pose import (
    umeyama_align_ransac,
    align_camera_pose,
    compute_alignment_error,
)


def rotation_angle_deg(R):
    """返回旋转矩阵对应的旋转角度（度）。"""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def average_rotations(R_list):
    """将一组旋转矩阵平均为一个旋转矩阵（四元数均值，含符号对齐）。"""
    qs = np.array([rotmat2qvec(R) for R in R_list])  # (N, 4) -> (w, x, y, z)
    ref = qs[0]
    for i in range(len(qs)):
        if np.dot(qs[i], ref) < 0:
            qs[i] = -qs[i]
    q_avg = qs.mean(axis=0)
    q_avg = q_avg / np.linalg.norm(q_avg)
    return qvec2rotmat(q_avg)


def snap_to_180(R):
    """把旋转矩阵 R 的旋转角度强制为精确 180°（旋转轴保持不变）。"""
    # 旋转轴 = R 的实特征值 +1 对应的特征向量（对 180° 附近最稳定）
    eigvals, eigvecs = np.linalg.eig(R)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    axis = np.real(eigvecs[:, idx])
    axis = axis / np.linalg.norm(axis)
    # 绕 axis 精确旋转 180°: R = 2 u u^T - I
    R_180 = 2.0 * np.outer(axis, axis) - np.eye(3)
    if np.linalg.det(R_180) < 0:
        R_180 = -R_180
    return R_180


def strip_ext(name):
    """去掉文件名后缀，用于跨模型匹配图像名称。"""
    return name.split(".")[0] if "." in name else name


def build_name_dicts(base_images, update_images):
    """把 image_id -> Image 的字典转换为 去后缀文件名 -> Image 的字典。"""
    base_by_name = {strip_ext(img.name): img for img in base_images.values()}
    update_by_name = {strip_ext(img.name): img for img in update_images.values()}
    return base_by_name, update_by_name


def estimate_alignment(base_by_name, update_by_name):
    """
    基于相机中心做 RANSAC umeyama 对齐（与 measure_pose.py 中一致）：
        update 相机中心 ~= scale * R_align @ base 相机中心 + shift
    """
    common = [n for n in base_by_name if n in update_by_name]
    if len(common) < 3:
        raise RuntimeError(f"匹配图像数量不足: {len(common)} (< 3)")

    base_centers = {}
    update_centers = {}
    for n in common:
        Rb = qvec2rotmat(base_by_name[n].qvec)
        Ru = qvec2rotmat(update_by_name[n].qvec)
        base_centers[n] = -Rb.T @ base_by_name[n].tvec
        update_centers[n] = -Ru.T @ update_by_name[n].tvec

    scale, R_align, shift, inliers = umeyama_align_ransac(
        base_centers,
        update_centers,
        max_iters=1000,
        inlier_threshold=2.0,  # 米
        min_inliers=max(3, int(0.5 * len(base_centers))),
    )
    return common, scale, R_align, shift


def transform_base_to_update_frame(base_by_name, scale, R_align, shift):
    """把 base 位姿变换到 update 坐标系（与 measure_pose.align_and_compute_error 一致）。"""
    base_aligned = {}
    for name, img in base_by_name.items():
        Rwc = qvec2rotmat(img.qvec)
        R_new, t_new, q_new = align_camera_pose(Rwc, img.tvec, R_align, scale, shift)
        base_aligned[name] = ColmapImage(
            id=img.id,
            qvec=q_new,
            tvec=t_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
    return base_aligned


def residual_rotation_errors(base_by_name, update_by_name, common, R_align):
    """
    计算每个匹配图像对的残差旋转：
        Rwc_base_aligned = Rwc_base @ R_align.T
        R_diff = Rwc_base_aligned @ Rwc_update.T
    """
    R_diffs, angles = [], []
    for n in common:
        Rb = qvec2rotmat(base_by_name[n].qvec)
        Ru = qvec2rotmat(update_by_name[n].qvec)
        R_diff = (Rb @ R_align.T) @ Ru.T
        R_diffs.append(R_diff)
        angles.append(rotation_angle_deg(R_diff))
    return R_diffs, np.array(angles)


def correct_update_pose(Rwc, twc, R_flip):
    """
    纠正单个相机位姿（保持相机中心不变）：
        C = -Rwc.T @ twc  保持不变
        Rwc' = R_flip @ Rwc
        twc' = -Rwc' @ C = R_flip @ twc
    """
    Rwc_new = R_flip @ Rwc
    twc_new = R_flip @ twc
    return Rwc_new, twc_new


def main():
    parser = ArgumentParser(
        description="纠正 update COLMAP 位姿相对 base 的 ~180° 旋转误差并写回"
    )
    parser.add_argument(
        "--colmap_output_base", "-b", type=str, required=True,
        help="base COLMAP sparse 模型目录",
    )
    parser.add_argument(
        "--colmap_output_update", "-u", type=str, required=True,
        help="update COLMAP sparse 模型目录（纠正后写回此处）",
    )
    parser.add_argument(
        "--flip_threshold", type=float, default=90.0,
        help="旋转误差中位数超过该角度（度）才认为存在翻转，默认 90",
    )
    parser.add_argument(
        "--flip_snap_tolerance", type=float, default=20.0,
        help="估计翻转角度与 180° 的偏差在容差（度）内时，强制使用精确 180° 旋转纠正（轴不变），默认 20",
    )
    parser.add_argument(
        "--visualize", "-vis", action="store_true", default=False,
        help="是否可视化纠正后的误差分布图",
    )
    parser.add_argument(
        "--no_backup", action="store_true", default=False,
        help="写回前不备份原始 update 模型",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.colmap_output_base) or not os.path.isdir(args.colmap_output_update):
        print("错误: base / update 目录不存在。")
        sys.exit(1)

    # ---------- 1. 读取模型 ----------
    base_cameras, base_images, base_points = read_model(args.colmap_output_base)
    update_cameras, update_images, update_points = read_model(args.colmap_output_update)
    if base_images is None or update_images is None:
        print("错误: 读取 COLMAP 模型失败。")
        sys.exit(1)
    print(f"base 图像数: {len(base_images)}, update 图像数: {len(update_images)}")

    # ---------- 2. 位置对齐 (base -> update frame) ----------
    base_by_name, update_by_name = build_name_dicts(base_images, update_images)
    common, scale, R_align, shift = estimate_alignment(base_by_name, update_by_name)
    print(f"匹配图像数量: {len(common)}")
    print(f"scale = {scale:.4f}")
    print(f"R_align =\n{R_align}")
    print(f"shift = {shift}")

    base_aligned = transform_base_to_update_frame(base_by_name, scale, R_align, shift)

    # ---------- 3. 计算旋转残差 ----------
    R_diffs, angles = residual_rotation_errors(base_by_name, update_by_name, common, R_align)
    median_angle = float(np.median(angles))
    print(
        f"旋转残差角度统计: median={median_angle:.2f}°, "
        f"mean={angles.mean():.2f}°, std={angles.std():.2f}°"
    )

    # ---------- 4. 判断是否需要翻转纠正 ----------
    if median_angle <= args.flip_threshold:
        print(
            f"中位旋转误差 {median_angle:.2f}° <= {args.flip_threshold}°，"
            "未检测到 ~180° 翻转，无需纠正。"
        )
        return

    flip_mask = angles > 90.0
    if int(flip_mask.sum()) < 3:
        print("旋转误差 >90° 的图像对不足 3 个，无法可靠估计翻转旋转。")
        return

    R_flip = average_rotations([R for R, m in zip(R_diffs, flip_mask) if m])
    print("\n检测到 ~180° 旋转翻转，估计的翻转旋转 R_flip:")
    print(R_flip)
    R_flip_angle = rotation_angle_deg(R_flip)
    print(f"R_flip 旋转角度: {R_flip_angle:.2f}°")

    # 旋转误差接近 180° 时，直接使用精确的 180° 旋转纠正（旋转轴保持不变）
    if abs(R_flip_angle - 180.0) <= args.flip_snap_tolerance:
        R_flip = snap_to_180(R_flip)
        print("旋转误差接近 180°，已强制使用精确 180° 旋转纠正（轴不变）:")
        print(R_flip)
    else:
        print(
            f"警告: 估计翻转角度 {R_flip_angle:.2f}° 偏离 180° 超过 "
            f"{args.flip_snap_tolerance}°，使用估计值纠正。"
        )

    # ---------- 5. 纠正 update 位姿 ----------
    corrected_images = {}
    corrected_by_name = {}
    for img_id, img in update_images.items():
        Rwc = qvec2rotmat(img.qvec)
        twc = img.tvec
        Rwc_new, twc_new = correct_update_pose(Rwc, twc, R_flip)
        new_img = ColmapImage(
            id=img.id,
            qvec=rotmat2qvec(Rwc_new),
            tvec=twc_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
        corrected_images[img_id] = new_img
        corrected_by_name[strip_ext(img.name)] = new_img

    # ---------- 6. 验证（同一对齐下，纠正前/后对比） ----------
    err_before = compute_alignment_error(base_aligned, update_by_name, visualize=False)
    err_after = compute_alignment_error(base_aligned, corrected_by_name, visualize=args.visualize)

    print("\n================ 纠正前 ================")
    print(
        f"  位置误差: mean={err_before.ate_error_mean:.3f}m, "
        f"rmse={err_before.ate_error_rmse:.3f}m"
    )
    print(
        f"  旋转误差: mean={err_before.rotate_angle_error_mean:.3f}°, "
        f"median={err_before.rotate_angle_error_median:.3f}°, "
        f"rmse={err_before.rotate_angle_error_rmse:.3f}°"
    )
    print("================ 纠正后 ================")
    print(
        f"  位置误差: mean={err_after.ate_error_mean:.3f}m, "
        f"rmse={err_after.ate_error_rmse:.3f}m"
    )
    print(
        f"  旋转误差: mean={err_after.rotate_angle_error_mean:.3f}°, "
        f"median={err_after.rotate_angle_error_median:.3f}°, "
        f"rmse={err_after.rotate_angle_error_rmse:.3f}°"
    )

    # ---------- 7. 备份并写回 ----------
    if not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{args.colmap_output_update}_backup_{ts}"
        shutil.copytree(args.colmap_output_update, backup_dir)
        print(f"\n已备份原始 update 模型到: {backup_dir}")

    ext = ".txt" if detect_model_format(args.colmap_output_update, ".txt") else ".bin"
    write_model(update_cameras, corrected_images, update_points, args.colmap_output_update, ext=ext)
    print(f"已写回纠正后的 update 位姿到: {args.colmap_output_update} (format={ext})")


if __name__ == "__main__":
    main()
