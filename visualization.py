
import os
import cv2
import numpy as np

def visualize_image_pairs(view_graph, images, image_path, output_dir, num_pairs=100):
    """
    从view_graph, cameras, images中读取数据，将每一对图像拼接在一起，
    并取出匹配的特征点，在特征点之间连线。
    
    Args:
        view_graph: 视图图对象
        images: 图像列表
        image_path: 图像文件路径
        num_pairs: 要可视化的图像对数量
    """
    # 创建保存可视化结果的目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像对
    image_pairs = list(view_graph.image_pairs.items())
    
    # 限制图像对数量
    # image_pairs = image_pairs[:num_pairs]
    
    for idx, (pair_id, image_pair) in enumerate(image_pairs):
        if not image_pair.is_valid:
            continue
            
        # 获取图像ID
        image_id1 = image_pair.image_id1
        image_id2 = image_pair.image_id2
        
        # 获取图像文件名
        image_name1 = images[image_id1].filename
        image_name2 = images[image_id2].filename
        
        # 构建完整路径
        img_path1 = os.path.join(image_path, image_name1)
        img_path2 = os.path.join(image_path, image_name2)
        
        # 检查文件是否存在
        if not os.path.exists(img_path1) or not os.path.exists(img_path2):
            continue
            
        # 读取图像
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            continue
            
        # 调整图像大小以便拼接
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        
        max_height = max(height1, height2)
        scale1 = max_height / height1 if height1 > 0 else 1
        scale2 = max_height / height2 if height2 > 0 else 1
        
        img1_resized = cv2.resize(img1, None, fx=scale1, fy=scale1)
        img2_resized = cv2.resize(img2, None, fx=scale2, fy=scale2)
        
        # 拼接图像
        combined_img = np.hstack((img1_resized, img2_resized))
        
        # 获取匹配点
        matches = image_pair.matches
        if matches is not None and len(matches) > 0:
            # 获取特征点
            keypoints1 = images[image_id1].features
            keypoints2 = images[image_id2].features
            
            # 从匹配点中随机选取50个，不要按顺序选
            num_matches_to_show = min(num_pairs, len(matches))
            # 随机选择索引
            random_indices = np.random.choice(len(matches), num_matches_to_show, replace=False)
            
            # 为每个匹配点对生成随机颜色
            for i in random_indices:
                match = matches[i]
                idx1, idx2 = match[0], match[1]
                
                if idx1 < len(keypoints1) and idx2 < len(keypoints2):
                    # 获取特征点坐标
                    pt1 = tuple(map(int, keypoints1[idx1] * scale1))
                    pt2 = tuple(map(int, keypoints2[idx2] * scale2))
                    
                    # 调整第二个图像的x坐标以适应拼接后的图像
                    pt2_shifted = (pt2[0] + img1_resized.shape[1], pt2[1])
                    
                    # 为每个匹配点对生成相同的随机颜色
                    color = tuple(map(int, np.random.randint(0, 255, 3)))
                    
                    # 在图像上绘制点和连线，使用相同的颜色
                    cv2.circle(combined_img, pt1, 3, color, -1)
                    cv2.circle(combined_img, pt2_shifted, 3, color, -1)
                    cv2.line(combined_img, pt1, pt2_shifted, color, 1)
        
        # 生成输出文件名：使用图像1和图像2的文件名（去除后缀后用_连接）
        name1 = os.path.splitext(image_name1)[0]
        name2 = os.path.splitext(image_name2)[0]
        output_filename = f"{name1}_{name2}_{len(matches)}_{image_pair.matches_init_num}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, combined_img)
        print(f"Saved visualization of image pair {idx} to {output_path}")


def draw_keypoints_on_image(image_path, keypoints, output_path=None, color=(0, 255, 0),
                            radius=3, thickness=-1, scores=None, max_points=None):
    """
    将特征点绘制在图像上。

    Args:
        image_path: 图像文件路径，或已加载的 numpy 图像 (H, W, 3)
        keypoints: 特征点坐标，shape 为 (N, 2)，每行为 (x, y)
        output_path: 输出图像保存路径，如果为 None 则返回图像数组
        color: 点的颜色，默认为绿色 (0, 255, 0)
        radius: 点的半径，默认 3
        thickness: 点的线宽，-1 表示实心圆
        scores: 可选，每个特征点的分数，用于颜色映射（高分红色，低分蓝色）
        max_points: 可选，最多绘制的点数，随机采样
    Returns:
        如果 output_path 为 None，返回绘制后的 numpy 图像数组
    """
    # 加载图像
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
    elif isinstance(image_path, np.ndarray):
        img = image_path.copy()
    else:
        raise TypeError("image_path 需为文件路径字符串或 numpy 数组")

    kpts = np.asarray(keypoints)
    if kpts.ndim != 2 or kpts.shape[1] != 2:
        raise ValueError(f"keypoints 需为 (N, 2) 形状，当前形状: {kpts.shape}")

    # 限制绘制点数
    if max_points is not None and max_points < len(kpts):
        indices = np.random.choice(len(kpts), max_points, replace=False)
        kpts = kpts[indices]
        if scores is not None:
            scores = np.asarray(scores)[indices]

    # 根据分数映射颜色
    if scores is not None:
        scores = np.asarray(scores)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-6:
            normalized = (scores - s_min) / (s_max - s_min)
        else:
            normalized = np.zeros_like(scores)
        # 低分蓝色 -> 高分红色
        for pt, norm in zip(kpts.astype(int), normalized):
            c = (int(255 * (1 - norm)), 0, int(255 * norm))
            cv2.circle(img, tuple(pt), radius, c, thickness)
    else:
        for pt in kpts.astype(int):
            cv2.circle(img, tuple(pt), radius, color, thickness)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"特征点图像已保存至: {output_path}")
        return None
    else:
        return img
