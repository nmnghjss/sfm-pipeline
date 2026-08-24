"""
将 source COLMAP 模型对齐到 target COLMAP 模型，并写出对齐后的模型。

参考 measure_pose.py 的对齐逻辑（基于相机中心的 RANSAC umeyama 相似变换）：
    target 相机中心 ~= scale * R_align @ source 相机中心 + shift

对 source 模型中的所有相机位姿和三维点做相同的相似变换，使整个模型
（相机位姿 + 三维点云）落入 target 的坐标系中。

对齐变换（缩放 + 旋转 + 平移）:
    P_new  = scale * (R_align @ P) + shift       # 三维点
    C_new  = scale * (R_align @ C) + shift       # 相机中心
    R_new  = Rwc @ R_align.T                     # 相机朝向
    t_new  = -R_new @ C_new                      # COLMAP 平移（w2c）

用法:
    python align_colmap_models.py --source ./model_a --target ./model_b \
        --output ./model_a_aligned

参数说明:
    --source  被对齐的 COLMAP 稀疏模型
    --target  作为参考坐标系的 COLMAP 稀疏模型
    --output  对齐后模型的输出目录（默认: <source>_aligned）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 允许从项目根目录导入 read_write_model / measure_pose 等模块
parent_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(parent_dir))

from read_write_model import (  # noqa: E402
    read_model,
    write_model,
    detect_model_format,
    qvec2rotmat,
    rotmat2qvec,
    Image as ColmapImage,
    Point3D,
)
from measure_pose import (  # noqa: E402
    umeyama_align_ransac,
    align_camera_pose,
    compute_alignment_error,
)


def strip_ext(name):
    """去掉文件名后缀，用于跨模型匹配图像名称。"""
    return name.split(".")[0] if "." in name else name


def build_name_dicts(images):
    """将 image_id -> Image 字典转换为 去后缀文件名 -> Image 字典。"""
    return {strip_ext(img.name): img for img in images.values()}


def collect_camera_centers(name_to_img, names):
    """按名称列表收集相机中心（COLMAP 约定：C = -R^T · t）。"""
    centers = {}
    for n in names:
        img = name_to_img[n]
        R = qvec2rotmat(img.qvec)
        centers[n] = -R.T @ img.tvec
    return centers


def estimate_alignment(source_by_name, target_by_name, inlier_threshold, max_iters, random_seed):
    """
    基于共享图像的相机中心，用 RANSAC umeyama 估计 source -> target 的相似变换。
    返回 (common_names, scale, R_align, shift)。
    """
    common = [n for n in source_by_name if n in target_by_name]
    if len(common) < 3:
        raise RuntimeError(f"匹配图像数量不足: {len(common)} (< 3)")

    source_centers = collect_camera_centers(source_by_name, common)
    target_centers = collect_camera_centers(target_by_name, common)

    scale, R_align, shift, inliers = umeyama_align_ransac(
        source_centers,
        target_centers,
        max_iters=max_iters,
        inlier_threshold=inlier_threshold,
        min_inliers=max(3, int(0.5 * len(common))),
        random_seed=random_seed,
    )
    return common, scale, R_align, shift


def transform_images(images, scale, R_align, shift):
    """对 source 所有相机位姿做相似变换，返回新的 image_id -> Image 字典。"""
    aligned = {}
    for img_id, img in images.items():
        Rwc = qvec2rotmat(img.qvec)
        R_new, t_new, q_new = align_camera_pose(Rwc, img.tvec, R_align, scale, shift)
        aligned[img_id] = ColmapImage(
            id=img.id,
            qvec=q_new,
            tvec=t_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
    return aligned


def transform_points(points3D, scale, R_align, shift):
    """对 source 所有三维点做相似变换，返回新的 point3D_id -> Point3D 字典。"""
    if not points3D:
        return points3D
    aligned = {}
    for pid, pt in points3D.items():
        aligned[pid] = Point3D(
            id=pt.id,
            xyz=scale * (R_align @ pt.xyz) + shift,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=pt.image_ids,
            point2D_idxs=pt.point2D_idxs,
        )
    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="将 source COLMAP 模型对齐到 target COLMAP 模型并写出"
    )
    parser.add_argument("--source", "-s", required=True,
                        help="被对齐的 COLMAP sparse 模型目录")
    parser.add_argument("--target", "-t", required=True,
                        help="作为参考坐标系的 COLMAP sparse 模型目录")
    parser.add_argument("--output", "-o", default="",
                        help="对齐后模型的输出目录（默认: <source>_aligned）")
    parser.add_argument("--inlier_threshold", type=float, default=2.0,
                        help="RANSAC 内点距离阈值（米），默认 2.0")
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="RANSAC 最大迭代次数，默认 1000")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="RANSAC 随机种子（便于复现），默认 None")
    parser.add_argument("--no_align_points", action="store_true", default=False,
                        help="不对三维点做变换（仅对齐相机位姿）")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="输出目录已存在且非空时仍覆盖写入")
    parser.add_argument("--visualize", "-vis", action="store_true", default=False,
                        help="可视化对齐后的误差分布图")
    args = parser.parse_args()

    # ---------- 1. 读取模型 ----------
    if not os.path.isdir(args.source) or not os.path.isdir(args.target):
        print("错误: source / target 目录不存在。", file=sys.stderr)
        sys.exit(1)

    src_cameras, src_images, src_points = read_model(args.source)
    tgt_cameras, tgt_images, _ = read_model(args.target)
    if src_images is None or tgt_images is None:
        print("错误: 读取 COLMAP 模型失败（缺少 cameras/images/points3D 的 .bin 或 .txt）。",
              file=sys.stderr)
        sys.exit(1)
    print(f"source 图像数: {len(src_images)}, target 图像数: {len(tgt_images)}")

    # ---------- 2. 估计对齐变换 ----------
    src_by_name = build_name_dicts(src_images)
    tgt_by_name = build_name_dicts(tgt_images)
    common, scale, R_align, shift = estimate_alignment(
        src_by_name, tgt_by_name,
        inlier_threshold=args.inlier_threshold,
        max_iters=args.max_iters,
        random_seed=args.random_seed,
    )
    print(f"匹配图像数量: {len(common)}")
    print(f"scale = {scale:.6f}")
    print(f"R_align =\n{R_align}")
    print(f"shift = {shift}")

    # ---------- 3. 应用变换 ----------
    aligned_images = transform_images(src_images, scale, R_align, shift)
    aligned_points = src_points if args.no_align_points else transform_points(
        src_points, scale, R_align, shift)

    # ---------- 4. 验证对齐误差（按名称对齐后与 target 对比） ----------
    aligned_by_name = build_name_dicts(aligned_images)
    err = compute_alignment_error(aligned_by_name, tgt_by_name, visualize=args.visualize)
    if err is not None:
        print("\n================ 对齐后误差 ================")
        print(f"  位置误差: mean={err.ate_error_mean:.3f}m, "
              f"rmse={err.ate_error_rmse:.3f}m, p90={err.ate_error_p90:.3f}m")
        print(f"  旋转误差: mean={err.rotate_angle_error_mean:.3f}°, "
              f"median={err.rotate_angle_error_median:.3f}°, "
              f"rmse={err.rotate_angle_error_rmse:.3f}°")

    # ---------- 5. 写出对齐后的模型 ----------
    output = args.output if args.output else f"{args.source}_aligned"
    if os.path.isdir(output) and os.listdir(output) and not args.overwrite:
        print(f"错误: 输出目录 {output} 非空，请使用 --output 指定新目录或加 --overwrite。",
              file=sys.stderr)
        sys.exit(1)
    os.makedirs(output, exist_ok=True)

    ext = ".txt" if detect_model_format(args.source, ".txt") else ".bin"
    write_model(src_cameras, aligned_images, aligned_points, output, ext=ext)
    print(f"\n已写出对齐后的模型到: {output} (format={ext})")


if __name__ == "__main__":
    main()
