import os
import time
import logging
import torch
from lightglue import LightGlue
from lightglue.utils import rbd
from database import COLMAPDatabase, batch_write_matches_to_database
from match_utils import load_image_pairs

# 20260630 add LOMA
from loma.loma import LoMa, filter_matches
from database import batch_write_keypoints_to_database
import numpy as np


def generate_sequential_match_list(
    image_names: list,
    overlap: int = 30,
    output_file: str = None,
    logger: logging.Logger = None
) -> list:
    """
    Generate sequential image matching pairs with circular overlap.
    
    This function creates a list of image pairs for sequential matching where:
    - Images are sorted and matched with their neighbors
    - Each image matches with N images before and after it (overlap parameter)
    - For images at the beginning, it loops to match with images at the end
    - For images at the end, it loops to match with images at the beginning
    
    Args:
        image_names: List of image names (will be sorted)
        overlap: Number of neighboring images to match on each side (default: 30)
        output_file: Optional path to save match list to text file
        logger: Optional logger instance
    
    Returns:
        List of tuples (image1_name, image2_name) representing match pairs
        Also writes to output_file if provided
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Sort image names for consistent ordering
    sorted_images = sorted(image_names)
    n_images = len(sorted_images)
    
    if n_images < 2:
        logger.warning("Not enough images for sequential matching")
        return []
    
    # Adjust overlap to not exceed available images
    effective_overlap = min(overlap, n_images - 1)
    
    match_pairs = []
    match_set = set()  # To avoid duplicate pairs
    
    for i in range(n_images):
        img0 = sorted_images[i]
        
        # Generate indices for images to match with (before and after)
        # Range: [i - effective_overlap, i + effective_overlap]
        match_indices = []
        
        # Add images after current image
        for offset in range(1, effective_overlap + 1):
            j = (i + offset) % n_images
            match_indices.append(j)
        
        # Add images before current image (to complete the circle)
        for offset in range(1, effective_overlap + 1):
            j = (i - offset) % n_images
            if j < i:  # Only add if not already added
                match_indices.append(j)
        
        # Create match pairs ensuring i < j to avoid duplicates
        for j in match_indices:
            if i != j:
                # Create canonical pair (smaller index first)
                pair_i, pair_j = (i, j) if i < j else (j, i)
                pair_key = (pair_i, pair_j)
                
                if pair_key not in match_set:
                    match_set.add(pair_key)
                    match_pairs.append((sorted_images[pair_i], sorted_images[pair_j]))
    
    logger.info(f"Generated {len(match_pairs)} sequential match pairs with overlap={effective_overlap}")
    
    # Write to file if specified
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                for img1, img2 in match_pairs:
                    f.write(f"{img1} {img2}\n")
            logger.info(f"Match list saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save match list to {output_file}: {e}")
    
    return match_pairs


def get_matches_importer_cmd(colmap_command: str, 
                             log_level: int, 
                             database_path: str, 
                             matched_images_pairs_path: str, 
                             feature_match_type: str, 
                             use_gpu: int = 1,
                             max_feature_num: int = 2048,
                             min_num_inliers: int = 30,
                             min_inlier_ratio: float = 0.1, 
                             two_view_geometry_max_error: float = 4.0,                            
                             sift_lightglue_match_path: str = "",
                             bruteforce_match_path: str = "",
                             aliked_lightglue_match_path: str = ""
):
    matches_importer_cmd = [
        colmap_command, "matches_importer",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--match_list_path", matched_images_pairs_path,
        "--match_type", "pairs", # {'pairs', 'raw', 'inliers'}               
        # "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT, ALIKED_LIGHTGLUE, ALIKED_N32
        "--FeatureMatching.num_threads", "-1",
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.guided_matching", "0",
        "--FeatureMatching.skip_geometric_verification", "0",
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.cpu_brute_force_matcher", "0",
        "--SiftMatching.lightglue_min_score", "0.1",
        "--SiftMatching.lightglue_model_path", sift_lightglue_match_path,
        "--AlikedMatching.brute_force_min_cossim", "0.85",
        "--AlikedMatching.brute_force_max_ratio", "1",
        "--AlikedMatching.brute_force_cross_check", "1",
        "--AlikedMatching.bruteforce_model_path", bruteforce_match_path,
        "--AlikedMatching.lightglue_min_score", "0.1",
        "--AlikedMatching.lightglue_model_path", aliked_lightglue_match_path,
        "--TwoViewGeometry.min_num_inliers", str(min_num_inliers), # 15
        "--TwoViewGeometry.multiple_models", "0",
        "--TwoViewGeometry.compute_relative_pose", "0",
        "--TwoViewGeometry.detect_watermark", "1",
        "--TwoViewGeometry.multiple_ignore_watermark", "1",
        "--TwoViewGeometry.watermark_detection_max_error", "4",
        "--TwoViewGeometry.filter_stationary_matches", "0",
        "--TwoViewGeometry.stationary_matches_max_error", "4",
        "--TwoViewGeometry.max_error", str(two_view_geometry_max_error),
        "--TwoViewGeometry.confidence", "0.999",
        "--TwoViewGeometry.max_num_trials", "10000",
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--TwoViewGeometry.random_seed", "-1"
    ]

    return matches_importer_cmd


def get_exhaustive_matcher_cmd(colmap_command: str, 
                             log_level: int, 
                             database_path: str, 
                             feature_match_type: str, 
                             use_gpu: int,
                             max_feature_num: int,
                             min_num_inliers: int,
                             min_inlier_ratio: float,
                             sift_lightglue_match_path: str,
                             bruteforce_match_path: str,
                             aliked_lightglue_match_path: str,
                             two_view_geometry_max_error: float=4.0):

    exhaustive_matcher_cmd = [
        colmap_command, "exhaustive_matcher",
        "--log_level", str(log_level), # 0
        "--database_path", database_path,
        "--ExhaustiveMatching.block_size", "200", # 50 
        "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT_BRUTEFORCE, ALIKED_LIGHTGLUE, ALIKED_N32
        # "--FeatureMatching.num_threads", str(args.num_threads),
        "--FeatureMatching.use_gpu", str(use_gpu),
        # "--FeatureMatching.gpu_index", str(args.gpu_index),
        "--FeatureMatching.guided_matching", "0",
        # "--FeatureMatching.skip_geometric_verification", str(args.skip_geometric_verification),
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.cpu_brute_force_matcher", "0",
        "--SiftMatching.lightglue_min_score", "0.1",
        "--SiftMatching.lightglue_model_path", sift_lightglue_match_path,
        "--AlikedMatching.brute_force_min_cossim", "0.85",
        "--AlikedMatching.brute_force_max_ratio", "1",
        "--AlikedMatching.brute_force_cross_check", "1",
        "--AlikedMatching.bruteforce_model_path", bruteforce_match_path,
        "--AlikedMatching.lightglue_min_score", "0.1",
        "--AlikedMatching.lightglue_model_path", aliked_lightglue_match_path,
        "--TwoViewGeometry.min_num_inliers", str(min_num_inliers), # 15
        "--TwoViewGeometry.multiple_models", "0",
        "--TwoViewGeometry.compute_relative_pose", "0",
        "--TwoViewGeometry.detect_watermark", "1",
        "--TwoViewGeometry.multiple_ignore_watermark", "1",
        "--TwoViewGeometry.watermark_detection_max_error", "4",
        "--TwoViewGeometry.filter_stationary_matches", "0",
        "--TwoViewGeometry.stationary_matches_max_error", "4",
        "--TwoViewGeometry.max_error", str(two_view_geometry_max_error),
        "--TwoViewGeometry.confidence", "0.999",
        "--TwoViewGeometry.max_num_trials", "10000",
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--TwoViewGeometry.random_seed", "-1",
    ]

    return exhaustive_matcher_cmd


def get_vocab_tree_matcher_cmd(colmap_command: str, 
                             log_level: int, 
                             database_path: str, 
                             feature_match_type: str, 
                             use_gpu: int,
                             max_feature_num: int,
                             min_num_inliers: int,
                             min_inlier_ratio: float,
                             two_view_geometry_max_error: float,
                             sift_lightglue_match_path: str,
                             bruteforce_match_path: str,
                             aliked_lightglue_match_path: str,
                             max_matches_per_image: int,
                             vocab_feature_num: int,
                             vocab_path: str):

    vocab_tree_matcher_cmd = [
        colmap_command, "vocab_tree_matcher",
        "--log_level", str(log_level),
        "--database_path", database_path,        
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.type", feature_match_type,
        "--FeatureMatching.num_threads", "-1",        
        "--FeatureMatching.guided_matching", "0",
        "--FeatureMatching.skip_geometric_verification", "0",
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.cpu_brute_force_matcher", "0",
        "--SiftMatching.lightglue_min_score", "0.1",
        "--SiftMatching.lightglue_model_path", sift_lightglue_match_path,
        "--AlikedMatching.brute_force_min_cossim", "0.85",
        "--AlikedMatching.brute_force_max_ratio", "1",
        "--AlikedMatching.brute_force_cross_check", "1",
        "--AlikedMatching.bruteforce_model_path", bruteforce_match_path,
        "--AlikedMatching.lightglue_min_score", "0.1",
        "--AlikedMatching.lightglue_model_path", aliked_lightglue_match_path,
        "--TwoViewGeometry.min_num_inliers", str(min_num_inliers), # 15
        "--TwoViewGeometry.multiple_models", "0",
        "--TwoViewGeometry.compute_relative_pose", "0",
        "--TwoViewGeometry.detect_watermark", "0",
        "--TwoViewGeometry.multiple_ignore_watermark", "1",
        "--TwoViewGeometry.watermark_detection_max_error", "4",
        "--TwoViewGeometry.filter_stationary_matches", "0",
        "--TwoViewGeometry.stationary_matches_max_error", "4",
        "--TwoViewGeometry.max_error", str(two_view_geometry_max_error),
        "--TwoViewGeometry.confidence", "0.9999",
        "--TwoViewGeometry.max_num_trials", "10000",
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--TwoViewGeometry.random_seed", "-1",
        "--VocabTreeMatching.num_images", str(max_matches_per_image), # 100
        "--VocabTreeMatching.num_nearest_neighbors", "5", # 5
        "--VocabTreeMatching.num_checks", "64",
        "--VocabTreeMatching.num_images_after_verification", "0", # 0
        "--VocabTreeMatching.max_num_features", str(vocab_feature_num),
        "--VocabTreeMatching.vocab_tree_path", vocab_path,
        # "--VocabTreeMatching.match_list_path", match_list_path,
        "--VocabTreeMatching.num_threads", "-1"
    ]
    
    return vocab_tree_matcher_cmd


def find_similar_images_parallel(image_features, max_num=30, min_num=20, threshold=None, logger=None):
    """
    计算图像相似度（并行优化），找到最相似的K张图或根据阈值找相似图。
    使用批量cosine_similarity计算，效率高。
    
    Args:
        image_features: Dict of image features
        k: 每张图找到的最相似图像数 (仅当threshold为None时使用)
        threshold: 相似度阈值(0~1)。如果指定，将找出所有相似度 > threshold 的图像对
        logger: Optional logger instance
    
    Returns:
        List of (i, img_name0, j, img_name1) tuples representing match pairs
    """
    image_list = list(image_features.keys())
    n_images = len(image_list)
    
    if n_images < 2:
        return []
    
    # 预计算所有图像的平均描述符
    desc_means = {}
    valid_images = []
    
    for img_name in image_list:
        if image_features[img_name] is not None:
            if isinstance(image_features[img_name], dict):
                desc = image_features[img_name]['descriptors'].mean(dim=1)  # (256,)
            else:
                desc = image_features[img_name][1].mean(dim=1)  # (256,)
            # print(f"Image {img_name} desc shape: {desc.shape}")
            desc_means[img_name] = desc
            valid_images.append(img_name)
    
    if len(valid_images) < 2:
        return []
    
    # 创建描述符矩阵
    logger.info(f"Creating descriptor tensor from {len(valid_images)} images")
    descriptors_list = [desc_means[name].unsqueeze(0) if desc_means[name].dim() == 1 else desc_means[name] 
                       for name in valid_images]
    
    descriptors_tensor = torch.cat(descriptors_list, dim=0)  # (m, 256)
    logger.info(f"descriptors_tensor shape after cat: {descriptors_tensor.shape}")
    
    # 计算相似度矩阵（使用归一化向量的矩阵乘法）
    # 确保 descriptors_tensor 是 (m, 256)
    if descriptors_tensor.dim() != 2:
        descriptors_tensor = descriptors_tensor.reshape(-1, descriptors_tensor.shape[-1])
    
    logger.info(f"descriptors_tensor shape before norm: {descriptors_tensor.shape}")
    norm_descs = torch.nn.functional.normalize(descriptors_tensor, p=2, dim=1)  # (m, 256)
    logger.info(f"norm_descs shape: {norm_descs.shape}")
    similarity_matrix = norm_descs @ norm_descs.t()  # (m, m) - cosine similarity
    logger.info(f"similarity_matrix shape: {similarity_matrix.shape}")
    
    # 为每张图找到最相似的K张或基于阈值找相似图
    match_pairs = []
    
    if threshold is not None:
        # 基于阈值的匹配
        logger.info(f"Using similarity threshold: {threshold}")
        for i in range(len(valid_images)):
            similarities = similarity_matrix[i]  # (m,) - 第i张图与所有图的相似度
            
            # 找出所有相似度大于阈值的索引（排除自己，自己的相似度为1）
            similar_mask = similarities > threshold
            similar_indices = torch.where(similar_mask)[0]
            logger.info(f"Image {i} has {len(similar_indices)} similar images above threshold {threshold}")
            if len(similar_indices) < min_num:                
                topk_vals, similar_indices = torch.topk(similarities, min(min_num+1, len(valid_images)))
                logger.info(f"  Not enough similar images above threshold, keeping top-{min_num} instead (found {len(similar_indices)})")

            # 如果大于阈值的匹配太多，只取前30个最相似的
            max_similar_num = max_num
            if len(similar_indices) > max_similar_num:
                # 获取这些索引对应的相似度值
                similar_sims = similarities[similar_indices]
                # 排序并取前30个最相似的
                top_vals, top_local_idx = torch.topk(similar_sims, min(max_similar_num, len(similar_sims)))
                similar_indices = similar_indices[top_local_idx]
                logger.info(f"  Keeping top-{max_similar_num} similar images (limited from {len(similar_sims)})")

            for j_local in similar_indices:
                j_local = int(j_local.item())
                if i == j_local:  # 跳过自己
                    continue
                
                img_name0 = valid_images[i]
                img_name1 = valid_images[j_local]
                
                # 找到原始image_list中的索引
                i_orig = image_list.index(img_name0)
                j_orig = image_list.index(img_name1)
                
                if i_orig < j_orig:  # 避免重复
                    match_pairs.append((img_name0, img_name1))
    else:
        # Top-K 匹配
        for i in range(len(valid_images)):
            similarities = similarity_matrix[i]  # (m,) - 第i张图与所有图的相似度
            # print(f"Image {i} similarities shape: {similarities.shape}")
            
            # 使用 torch.topk 获取top-K相似的索引（排除自己）
            topk_vals, topk_indices = torch.topk(similarities, min(min_num+1, len(valid_images)))
            # print(f"topk_indices shape: {topk_indices.shape}")
            logger.info(f"topk_vals: {topk_vals}")
            
            # 跳过第一个（自己），取后面的k个
            for idx_in_topk in range(1, min(min_num+1, len(topk_indices))):
                j_local = int(topk_indices[idx_in_topk].item())  # 转换为Python整数
                
                img_name0 = valid_images[i]
                img_name1 = valid_images[j_local]
                
                # 找到原始image_list中的索引
                i_orig = image_list.index(img_name0)
                j_orig = image_list.index(img_name1)
                
                if i_orig < j_orig:  # 避免重复
                    match_pairs.append((img_name0, img_name1))
    
    if logger:
        if threshold is not None:
            logger.info(f"Found {len(match_pairs)} image pairs using similarity threshold ({threshold})")
        else:
            logger.info(f"Found {len(match_pairs)} image pairs using parallel similarity search (top-{k})")
    
    return match_pairs


def match_features_with_lightglue(
    weights_root: str,
    feature_type:str,
    database_path: str,
    image_features: dict,
    image_id_map: dict,
    match_list_path: str = None,
    match_strategy: str = "nearest_k",
    max_matches_per_image: int = 30,
    min_matches_per_image: int = 20,
    similarity_threshold: float = None,
    logger: logging.Logger = None
) -> tuple:
    """
    Match features using LightGlue and write to COLMAP database.
    
    Args:
        database_path: Path to COLMAP database
        image_features: Dict of image name -> feature tensor
        image_id_map: Dict of image name -> image_id in database
        match_list_path: Optional path to save match list
        match_strategy: 'exhaustive', 'nearest_k', 'threshold', or 'quick'
        max_matches_per_image: Max similar images per image (for nearest_k/quick)
        similarity_threshold: Similarity threshold for threshold-based matching
        logger: Optional logger instance
    
    Returns:
        feature_matching_time
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Initialize LightGlue matcher
    logger.info("Initializing LightGlue matcher...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = f"checkpoints/pth/{feature_type}_lightglue_v0-1_arxiv.pth"
    weight_path = os.path.join(weights_root, weight_path)
    logger.info(f"weight_path: {weight_path}")
    matcher = LightGlue(features=feature_type, local_path=weight_path).eval().to(device)

    logger.info(f"LightGlue Using device: {device}")
    # Connect to database
    try:
        db = COLMAPDatabase.connect(database_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 0
    
    # Feature matching
    logger.info("Matching features with LightGlue...")
    matching_start = time.time()
    
    image_list = [img for img in image_features.keys() if image_features[img] is not None]
    total_match_pairs = 0
    total_matches = 0
    match_statistics = []
    # img_num = len(image_list)
    # max_matches_per_image = int(min(max_matches_per_image, img_num * 0.3))
    
    prior_match_pairs = None
    if os.path.exists(match_list_path):
        prior_match_pairs = load_image_pairs(match_list_path)
        prior_match_pairs = set(prior_match_pairs)
        logger.info(f"Loaded match pairs from existing file: {match_list_path}, total pairs: {len(prior_match_pairs)}")

    # 根据策略选择匹配对
    if match_strategy == "exhaustive":
        match_pairs = []
        for i, img_name0 in enumerate(image_list):
            for j, img_name1 in enumerate(image_list):
                if i < j:
                    match_pairs.append((img_name0, img_name1))                   
        logger.info(f"Exhaustive matching: {len(match_pairs)} image pairs will be matched")
    
    elif match_strategy == "nearest_k":
        logger.info(f"Finding top-{max_matches_per_image} similar images for each (parallel)...")
        similarity_start = time.time()
        match_pairs = find_similar_images_parallel(image_features, max_num=max_matches_per_image, min_num=min_matches_per_image, threshold=None, logger=logger)
        logger.info(f"Similarity computation took {time.time() - similarity_start:.2f}s ({len(match_pairs)} pairs)")
    
    elif match_strategy == "threshold":
        logger.info(f"Using similarity threshold matching (threshold={similarity_threshold})...")
        similarity_start = time.time()
        match_pairs = find_similar_images_parallel(image_features, max_num=max_matches_per_image, min_num=min_matches_per_image, threshold=similarity_threshold, logger=logger)
        logger.info(f"Similarity computation took {time.time() - similarity_start:.2f}s ({len(match_pairs)} pairs)")

    else:
        logger.warning(f"Unknown match_strategy '{match_strategy}', falling back to exhaustive")
        match_pairs = []
        for i, img_name0 in enumerate(image_list):
            for j, img_name1 in enumerate(image_list):
                if i < j:
                    match_pairs.append((img_name0, img_name1))

    if prior_match_pairs is not None:
        filted_num = 0
        for img_name0, img_name1 in match_pairs:
            if (img_name0, img_name1) not in prior_match_pairs and (img_name1, img_name0) not in prior_match_pairs:
                match_pairs.remove((img_name0, img_name1))
                filted_num += 1
                logger.info(f"Filtering out pair ({img_name0}, {img_name1}) not in prior match pairs")
        logger.info(f"After filtering with prior match pairs, {len(match_pairs)} pairs remain for matching, filtered out {filted_num} pairs")
        # match_pairs = prior_match_pairs

    # Batch write match list to file if path provided
    try:
        os.makedirs(os.path.dirname(match_list_path), exist_ok=True)
        with open(match_list_path, 'w') as match_file:
            for img_name0, img_name1 in match_pairs:
                match_file.write(f"{img_name0} {img_name1}\n")
        logger.info(f"Match list written to {match_list_path}")
    except Exception as e:
        logger.warning(f"Could not write match_list_path file: {e}")

    # Buffer for batch write operations
    matches_to_write = []
    match_list_pairs = []
    
    # Perform matching
    logger.info(f"Performing LightGlue matching on {len(match_pairs)} pairs...")
    match_progress_interval = max(1, len(match_pairs) // 100)
    
    for pair_idx, (img_name0, img_name1) in enumerate(match_pairs):
        if pair_idx % match_progress_interval == 0 and pair_idx > 0:
            progress_pct = ((pair_idx+1) / len(match_pairs)) * 100
            logger.info(f"  Progress: {progress_pct:.1f}% ({pair_idx}/{len(match_pairs)})")
        
        try:
            feats0 = image_features[img_name0]
            feats1 = image_features[img_name1]
            
            if feats0 is None or feats1 is None:
                continue
            
            image_id0 = image_id_map[img_name0]
            image_id1 = image_id_map[img_name1]
            
            if image_id0 is None or image_id1 is None:
                continue
            
            # Perform matching
            with torch.no_grad():
                matches01 = matcher({'image0': feats0, 'image1': feats1})
            
            # Remove batch dimension
            feats0_rbd, feats1_rbd, matches01_rbd = [rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01_rbd['matches'].cpu().numpy()
            
            if len(matches) > 15:
                total_match_pairs += 1
                total_matches += len(matches)
                match_statistics.append((img_name0, img_name1, len(matches)))
                
                # Buffer matches for batch write
                matches_to_write.append((image_id0, image_id1, matches))
                match_list_pairs.append((img_name0, img_name1))
            else:
                logger.info(f"Pair ({img_name0}, {img_name1}) has only {len(matches)} matches, skipping")
                
        except Exception as e:
            logger.warning(f"Failed to match {img_name0} and {img_name1}: {e}")
            continue
    
    # Batch write all matches to database (optimized - single transaction)
    logger.info(f"Writing {total_match_pairs} match pairs to database in batch mode...")
    write_start = time.time()
    written_pairs = batch_write_matches_to_database(db, matches_to_write, logger)
    write_time = time.time() - write_start
    logger.info(f"Batch write completed in {write_time:.2f}s ({written_pairs} pairs written)")
    
    feature_matching_time = time.time() - matching_start
    logger.info(f"Feature matching by lightglue completed in {feature_matching_time:.2f}s")
    logger.info(f"  Matched {total_match_pairs} image pairs")
    logger.info(f"  Total matches found: {total_matches}")
    if total_match_pairs > 0:
        logger.info(f"  Average matches per pair: {total_matches/total_match_pairs:.1f}")
    logger.info("features and matches written to database")
    
    db.close()
    
    return feature_matching_time

def match_features_with_loma(
    weights_root: str,
    feature_type: str,  # "loma_B128", "loma_B", "loma_L", "loma_G", "loma_R"
    database_path: str,
    image_paths: list,
    match_list_path: str = None,
    match_strategy: str = "exhaustive",
    max_matches_per_image: int = 30,
    min_matches_per_image: int = 20,
    similarity_threshold: float = None,
    num_keypoints: int = 2048,
    descriptor_type_id: int = 3,           # ★ 新增：LOMA 在 COLMAP descriptors 表中的 type 编码
    logger: logging.Logger = None
) -> tuple:
    """
    Match features using LoMa and write to COLMAP database.
    
    Args:
        weights_root: Path to LoMa checkpoint directory
        feature_type: LoMa model type ("loma_B128", "loma_B", "loma_L", "loma_G", "loma_R")
        database_path: Path to COLMAP database
        image_paths: List of image file paths
        match_list_path: Optional path to save match list
        match_strategy: 'exhaustive', 'nearest_k', 'threshold', or 'quick'
        max_matches_per_image: Max similar images per image
        min_matches_per_image: Min similar images per image
        similarity_threshold: Similarity threshold for threshold-based matching
        num_keypoints: Number of keypoints to detect per image
        logger: Optional logger instance
    
    Returns:
        feature_matching_time
    """
    
    if logger is None:
        logger = logging.getLogger()
    
    # === 1. 初始化 LoMa 模型 ===
    logger.info(f"Initializing LoMa model: {feature_type}...")
    
    # 根据 feature_type 选择模型配置
    model_configs = {
        "loma_B128": ("checkpoint/loma_B128.pth", "dedode_b", 128),
        "loma_B": ("checkpoint/loma_B.pth", "dedode_g", 256),
        "loma_L": ("checkpoint/loma_L.pth", "dedode_g", 256),
        "loma_G": ("checkpoint/loma_G.pth", "dedode_g", 256),
        "loma_R": ("checkpoint/loma_R.pth", "dedode_g", 256),
    }
    
    if feature_type not in model_configs:
        raise ValueError(f"Unsupported LoMa model: {feature_type}")
    
    weights_path, descriptor_type, input_dim = model_configs[feature_type]
    weights_path = os.path.join(weights_root, weights_path)
    
    # 创建 LoMa 配置并加载模型
    cfg = LoMa.Cfg(
        descriptor=descriptor_type,
        input_dim=input_dim,
        weights_url=weights_path,
    )
    model = LoMa(cfg).eval()
    logger.info(f"LoMa model loaded from: {weights_path}")
    
    # === 2. 提取所有图像的特征 ===
    logger.info(f"Extracting features from {len(image_paths)} images...")
    extract_start = time.time()
    
    image_features = {}  # 存储 (kpts, desc, h, w) 元组
    valid_image_names = []
    
    def to_pixel_coords(flow, h1, w1):
        flow = torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            dim=-1,
        )
        return flow
    
    for img_path in image_paths:
        img_path_str = str(img_path)
        img_name = os.path.basename(img_path)
        try:
            kpts_, desc, h, w = model.detect_and_describe(img_path_str, num_keypoints=num_keypoints)
            kpts = to_pixel_coords(kpts_[0], h, w)
            image_features[img_name] = (kpts, desc, h, w)
            valid_image_names.append(img_name)
        except Exception as e:
            logger.warning(f"Failed to extract features from {img_name}: {e}")
            image_features[img_name] = None
    
    extract_time = time.time() - extract_start
    logger.info(f"Feature extraction completed in {extract_time:.2f}s")
    
    # === 3. 连接数据库 ===
    try:
        db = COLMAPDatabase.connect(database_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 0
    
    # === 4. 特征匹配 ===
    matching_start = time.time()
    total_match_pairs = 0
    total_matches = 0
    
    # 构建相似度矩阵（基于描述符均值）
    logger.info("Computing similarity matrix...")
    desc_means = {}
    valid_images = []
    
    for img_name in valid_image_names: # valid_image_names 是所有图片的名称列表0-n
        if image_features[img_name] is not None:
            kpts, desc, h, w = image_features[img_name]
            desc_mean = desc.mean(dim=1)  # (D,)
            desc_means[img_name] = desc_mean
            valid_images.append(img_name)
    
    # 构建相似度矩阵并选择匹配对
    if len(valid_images) < 2:
        logger.warning("Not enough valid images for matching")
        return 0
    
    desc_tensor = torch.stack([desc_means[n] for n in valid_images])
    norm_desc = torch.nn.functional.normalize(desc_tensor, p=2, dim=1)
    # similarity_matrix = norm_desc @ norm_desc.t()
    
    # 选择匹配策略
    match_pairs = []
    if match_strategy == "exhaustive":
        for i in range(len(valid_images)):
            for j in range(i + 1, len(valid_images)):
                match_pairs.append((valid_images[i], valid_images[j]))
    # elif match_strategy == "nearest_k":
    #     for i in range(len(valid_images)):
    #         sims = similarity_matrix[i]
    #         _, topk_idx = torch.topk(sims, min(min_matches_per_image + 1, len(valid_images)))
    #         for idx in topk_idx.tolist():
    #             if idx != i:
    #                 j = valid_images[idx]
    #                 img_i = valid_images[i]
    #                 if img_i < j:
    #                     match_pairs.append((img_i, j))
    #                 else:
    #                     match_pairs.append((j, img_i))
    elif match_strategy == "threshold":
        logger.info(f"Using similarity threshold matching (threshold={similarity_threshold})...")
        similarity_start = time.time()
        match_pairs = find_similar_images_parallel(image_features, max_num=max_matches_per_image, min_num=min_matches_per_image, threshold=similarity_threshold, logger=logger)
        logger.info(f"Similarity computation took {time.time() - similarity_start:.2f}s ({len(match_pairs)} pairs)")
    else:
        match_pairs = list(set(match_pairs))  # 去重
    
    # === 5. 执行匹配 ===
    logger.info(f"Performing LoMa matching on {len(match_pairs)} pairs...")

    image_id_map = {name: i + 1 for i, name in enumerate(valid_image_names)}
    matches_to_write = []

    for img_name0, img_name1 in match_pairs:
        try:
            if image_features[img_name0] is None or image_features[img_name1] is None:
                continue
            
            kpts0, desc0, h0, w0 = image_features[img_name0]
            kpts1, desc1, h1, w1 = image_features[img_name1]
            
            # 执行匹配
            with torch.inference_mode():
                scores = model(kpts0, kpts1, desc0, desc1)["scores"]
            
            # 提取有效匹配
            m0, *_ = filter_matches(scores, model.cfg.filter_threshold)
            valid = m0[0] > -1
            
            if valid.sum() < 15:
                continue
            
            # 提取有效匹配对的 keypoint indices（不是像素坐标！）
            valid_indices_0 = torch.where(valid)[0]   # image0 上的有效 keypoint 索引
            valid_indices_1 = m0[0][valid]            # image1 上对应的 keypoint 索引

            # print(f"valid_indices_0: {valid_indices_0}, valid_indices_1: {valid_indices_1}")

            # 构建匹配数组 [N x 2] = (keypoint_idx_in_img0, keypoint_idx_in_img1)
            matches = torch.stack([valid_indices_0, valid_indices_1], dim=1).cpu().numpy()

            image_id0 = image_id_map.get(img_name0)
            image_id1 = image_id_map.get(img_name1)
            
            if image_id0 is None or image_id1 is None:
                continue
            
            total_match_pairs += 1
            total_matches += len(matches)
            matches_to_write.append((image_id0, image_id1, matches))
            
        except Exception as e:
            logger.warning(f"Failed to match {img_name0} and {img_name1}: {e}")
            continue
    
    # === 6. 写入 image_pairs.txt（供 matches_importer 使用）===
    if match_list_path and len(match_pairs) > 0:
        try:
            os.makedirs(os.path.dirname(match_list_path) or ".", exist_ok=True)
            with open(match_list_path, "w") as f:
                for a, b in match_pairs:
                    f.write(f"{a} {b}\n")
            logger.info(f"Match list saved to: {match_list_path} ({len(match_pairs)} pairs)")
        except Exception as e:
            logger.error(f"Failed to write match list: {e}")

    # === 7. 批量写入 keypoints / descriptors 到数据库（mapper 必需）===
    logger.info("Writing keypoints/descriptors to database for mapper...")
    keypoints_to_write = []
    for img_name, feat in image_features.items():
        if feat is None:
            continue
        kpts, desc, h, w = feat
        image_id = image_id_map.get(img_name)
        if image_id is None:
            continue
        # kpts 形状 (1, N, 2) -> (N, 2)，desc 形状 (1, D, N) -> (D, N).T -> (N, D)
        kpts_np = kpts.squeeze(0).cpu().numpy().astype(np.float32) if hasattr(kpts, "cpu") else np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
        desc_np_ = desc.squeeze(0).T.cpu().numpy().astype(np.float32) if hasattr(desc, "cpu") else np.asarray(desc, dtype=np.float32).T.reshape(-1, desc.shape[1] if desc.ndim >= 1 else 128)
        desc_np = desc_np_.T
        desc_np = desc_np[:, :128]
        keypoints_to_write.append((image_id, kpts_np, desc_np))
        # input(f"Press Enter to continue...")
        print(f"image_id: {image_id}, kpts_np: {kpts_np.shape}, desc_np: {desc_np.shape}")
    written_kpts = batch_write_keypoints_to_database(
        db, keypoints_to_write, feature_type=descriptor_type_id, logger=logger
    )
    logger.info(f"Wrote keypoints for {written_kpts} images (descriptor_type={descriptor_type_id})")

    # === 8. 批量写入 matches 到数据库 ===
    logger.info(f"Writing {total_match_pairs} match pairs to database...")
    write_start = time.time()
    written_pairs = batch_write_matches_to_database(db, matches_to_write, logger)
    write_time = time.time() - write_start
    feature_matching_time = time.time() - matching_start
    total_time = time.time() - extract_start

    logger.info(f"Batch matches write completed in {write_time:.2f}s ({written_pairs} pairs written)")
    logger.info(f"Total time: {total_time:.2f}s (extract: {extract_time:.2f}s, match: {feature_matching_time:.2f}s)")
    logger.info(f"Matched {total_match_pairs} image pairs with {total_matches} total matches")

    db.close()
    return extract_time, feature_matching_time