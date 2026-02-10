import numpy as np
import argparse
# from joblib import delayed, Parallel
import json
import os
import time
from read_write_model import *
# from read_write_model import read_model
import torch

def get_visible_points_depth(key, cameras, images, points3d_ordered, points3d_errors_ordered, pmin=5, pmax=95):
    """
    获取图像中可见特征点的加权信息和逆深度
    
    Args:
        key: 图像键值
        cameras: 相机参数字典
        images: 图像元数据字典
        points3d_ordered: 有序的3D点数组
        points3d_errors_ordered: 有序的3D点误差数组，如果为空数组则不计算权重
    
    Returns:
        dict: 包含以下键值的字典
            - invcolmapdepth: 逆深度数组，如果valid_indices为0则返回None
            - visible_xys: 可见特征点的2D坐标
            - weights: 权重数组（如果points3d_errors_ordered为空则为None）
            - image_name: 图像名称
    """
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # 获取该图像的特征点索引
    pts_idx = images[key].point3D_ids
    
    # 只保留在该图像中可见的特征点（point3D_ids > -1）
    visible_mask = pts_idx >= 0
    visible_mask *= pts_idx < len(points3d_ordered)
    
    # 应用可见性掩码，只保留可见的特征点
    visible_pts_idx = pts_idx[visible_mask]
    visible_xys = image_meta.xys[visible_mask]
    
    print(f"图像 {image_meta.name}: 总特征点数={len(pts_idx)}, 可见特征点数={len(visible_pts_idx)}")
    
    if len(visible_pts_idx) == 0:
        print(f"警告: 图像 {image_meta.name} 没有可见的特征点")
        return None, None

    # 获取3D点坐标
    pts = points3d_ordered[visible_pts_idx]
    
    # 检查是否需要计算权重
    if len(points3d_errors_ordered) > 0:
        # 获取重投影误差作为权重
        # 直接从points3d_errors_ordered数组中获取error信息
        weights = []
        valid_indices = []
        
        for i, pt_idx in enumerate(visible_pts_idx):
            if pt_idx < len(points3d_errors_ordered) and points3d_errors_ordered[pt_idx] > 0:
                error = points3d_errors_ordered[pt_idx]
                if error > 1:
                    # print(f"警告: 点ID {pt_idx} 的重投影误差过大: {error:.4f}")
                    continue
                # 使用重投影误差的倒数作为权重，添加小的常数避免除零
                weight = 1.0 / (error + 1e-6)
                weights.append(weight)
                valid_indices.append(i)
            else:
                print(f"警告: 点ID {pt_idx} 的误差信息无效")
        
        if len(valid_indices) == 0:
            print(f"警告: 图像 {image_meta.name} 没有有效的权重信息")
            return None, None

        # 只保留有权重信息的点
        pts = pts[valid_indices]
        visible_xys = visible_xys[valid_indices]
        weights = np.array(weights)
        
        # print(f"有效加权点数: {len(weights)}, 权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
    else:
        # 不计算权重，使用所有可见点
        weights = None
        print(f"不使用权重，直接使用所有可见点数: {len(visible_pts_idx)}")

    # 将3D点转换到相机坐标系
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    # 计算深度
    depths = pts[..., 2]
    dmin = np.percentile(depths, pmin)
    dmax = np.percentile(depths, pmax)    
    return dmin, dmax


def estimate_depth_range(sparse_dir, pmin=5, pmax=95):
    """
    估计深度图的全局深度范围（百分位数法）
    
    Args:
        sparse_dir: COLMAP稀疏模型目录
        pmin: 最小百分位数
        pmax: 最大百分位数
    
    Returns:
        tuple: (depth_min, depth_max)
    """


    cam_intrinsics, images_metas, points3d = read_model(sparse_dir, ext=".bin")
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    pts_errors = np.array([points3d[key].error for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs
    points3d_errors_ordered = np.zeros(pts_indices.max()+1)
    points3d_errors_ordered[pts_indices] = pts_errors

    depth_values = []
    global_dmin = float('inf')
    global_dmax = float('-inf')

    for key in images_metas:
        image_meta = images_metas[key]
        cam_intrinsic = cam_intrinsics[image_meta.camera_id]

        pts_idx = images_metas[key].point3D_ids
        visible_mask = pts_idx >= 0
        visible_pts_idx = pts_idx[visible_mask]

        if len(visible_pts_idx) == 0:
            continue

        dmin, dmax = get_visible_points_depth(
            key, cam_intrinsics, images_metas, points3d_ordered, points3d_errors_ordered, pmin, pmax
        )
        print(f"图像 {image_meta.name}: 深度范围估计 [{dmin:.4f}, {dmax:.4f}]")
        if dmin is None or dmax is None:
            continue
        global_dmin = min(global_dmin, dmin)
        global_dmax = max(global_dmax, dmax)

    print(f"估计的深度范围: [{global_dmin:.4f}, {global_dmax:.4f}] (基于第{pmin}和{pmax}百分位数)")

    return global_dmin, global_dmax

