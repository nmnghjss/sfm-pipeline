import os
import time
import logging
import torch
from lightglue import LightGlue
from lightglue.utils import rbd
from database import COLMAPDatabase, batch_write_matches_to_database
from match_utils import load_image_pairs


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
            desc = image_features[img_name]['descriptors'].mean(dim=1)  # (256,)
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

