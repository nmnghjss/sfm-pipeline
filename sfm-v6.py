
import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from utils import *
from pathlib import Path
import torch
import numpy as np
import cv2
import sqlite3
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from database import COLMAPDatabase
# --------------------------
# Argument parser
# --------------------------
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", default="E:\\Test1234\\data18", type=str)
parser.add_argument("--output_path", "-o", default="output_splg", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--single_camera", "-sc",default="1", type=str)
parser.add_argument("--single_fold", "-sf", default="0", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--alg", default="acc", type=str, help="Algorithm for matching and mapping: colmap / acc / glomap")
parser.add_argument("--max_feature_num", default=2048, type=int, help="Maximum number of features to extract per image (for SuperPoint)")
parser.add_argument("--SuperpointLightglue", "-splg", action="store_true", help="Use SuperPoint features instead of SIFT")
parser.add_argument("--match_strategy", "-ms", type=str, default="threshold", 
                    choices=["exhaustive", "nearest_k", "quick", "threshold"],
                    help="Matching strategy: exhaustive (all pairs), nearest_k (top-k similar), quick (fast heuristic), threshold (similarity threshold based)")
parser.add_argument("--max_matches_per_image", "-mpi", type=int, default=30,
                    help="Max number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--similarity_threshold", "-st", type=float, default=0.75,
                    help="Similarity threshold for threshold-based matching strategy (0~1)")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
args = parser.parse_args()


# =====================key parameter =====================================================
min_num_inliers = 30
min_inlier_ratio = 0.8

# --------------------------
# Helper functions
# --------------------------
def resource_path() -> str:
    """Get absolute path for packaged or development scripts."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

# --------------------------
# Paths and executables
# --------------------------
current_path = resource_path()
os_type = check_operating_system()
print(f"Detected operating system: {os_type}")
if os_type == 'Windows':
    # colmap_path = os.path.join(current_path, "colmap-x64-windows-cuda-3.13.0/bin/colmap.exe")
    # colmap_path = "D:\\Codes\\Work\\colmap\\build\\src\\colmap\\exe\\Release\\colmap.exe"
    colmap_path = "D:\\Programs\\colmap-x64-windows-cuda-3.13.0\\bin\\colmap.exe"
    glomap_path = os.path.join(current_path, "D:\\Programs\\glomap-x64-windows-cuda-1.2.0\\bin\\glomap.exe")
else:
    colmap_path = "colmap"
    glomap_path = "glomap"
colmap_command = args.colmap_executable if args.colmap_executable else colmap_path
glomap_command = args.glomap_executable if args.glomap_executable else glomap_path

use_gpu = 0 if args.no_gpu else 1

if args.output_path == "":
    output_path = args.source_path
else:
    output_path = args.output_path
    if not Path(output_path).is_absolute():
        output_path = os.path.join(args.source_path, output_path)
os.makedirs(output_path, exist_ok=True)

# --------------------------
# Logging setup
# --------------------------
log_level = args.log_level
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger()
logger.setLevel(log_level)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(log_level)
ch.setFormatter(log_formatter)
logger.addHandler(ch)

# File handler
log_dir = output_path
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "run-sfm.log")
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(log_level)
fh.setFormatter(log_formatter)
logger.addHandler(fh)


# --------------------------
# Timing variables
# --------------------------
start_time = time.time()
feature_extraction_time = 0
feature_matching_time = 0
splg_time = 0
mapper_time = 0

# --------------------------
# SuperPoint + LightGlue Functions
# --------------------------

def write_keypoints_to_database(db, image_id: int, keypoints: np.ndarray, descriptors: np.ndarray, logger=None):
    """
    Write SuperPoint keypoints and descriptors to COLMAP database.
    Replaces COLMAP SIFT features with SuperPoint features.
    
    Args:
        db: COLMAPDatabase connection
        image_id: Image ID in database
        keypoints: (N, 2) array of keypoint coordinates
        descriptors: (N, 256) array of feature descriptors (SuperPoint outputs 256-dim)
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Delete existing features for this image first
        cursor = db.cursor()
        cursor.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
        cursor.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
        db.commit()
        
        # Ensure correct data types
        keypoints = np.asarray(keypoints, np.float32)
        descriptors = np.asarray(descriptors, np.uint8)  # COLMAP expects uint8
        
        # Use database methods to write
        db.add_keypoints(image_id, keypoints)
        db.add_descriptors(image_id, descriptors)
        db.commit()
        
        # logger.debug(f"Wrote {len(keypoints)} keypoints for image_id {image_id}")
        
    except Exception as e:
        logger.warning(f"Failed to write features to database for image {image_id}: {e}")


def write_matches_to_database(db, image_id0: int, image_id1: int, matches: np.ndarray, logger=None):
    """
    Write LightGlue matches to COLMAP database.
    
    Args:
        db: COLMAPDatabase connection
        image_id0: First image ID
        image_id1: Second image ID
        matches: (K, 2) array of keypoint indices that match
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Ensure image_id0 < image_id1 for consistent pair_id encoding
        if image_id0 > image_id1:
            image_id0, image_id1 = image_id1, image_id0
            matches = matches[:, [1, 0]]  # Swap match indices accordingly
        
        # Delete existing matches for this pair first (important!)
        cursor = db.cursor()
        MAX_IMAGE_ID = 2**31 - 1
        pair_id = image_id0 * MAX_IMAGE_ID + image_id1
        cursor.execute("DELETE FROM matches WHERE pair_id = ?", (pair_id,))
        db.commit()
        
        # Ensure correct data type
        matches = np.asarray(matches, np.uint32)
        
        # Use database method to write (handles pair_id encoding automatically)
        db.add_matches(image_id0, image_id1, matches)
        db.commit()
        
        # logger.debug(f"Wrote {len(matches)} matches between images {image_id0} and {image_id1}")
        
    except Exception as e:
        logger.warning(f"Failed to write matches to database for image pair ({image_id0}, {image_id1}): {e}")


def find_similar_images_parallel(image_features, k=10, threshold=None, logger=None):
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
    print(f"Creating descriptor tensor from {len(valid_images)} images")
    descriptors_list = [desc_means[name].unsqueeze(0) if desc_means[name].dim() == 1 else desc_means[name] 
                       for name in valid_images]
    
    descriptors_tensor = torch.cat(descriptors_list, dim=0)  # (m, 256)
    print(f"descriptors_tensor shape after cat: {descriptors_tensor.shape}")
    
    # 计算相似度矩阵（使用归一化向量的矩阵乘法）
    # 确保 descriptors_tensor 是 (m, 256)
    if descriptors_tensor.dim() != 2:
        descriptors_tensor = descriptors_tensor.reshape(-1, descriptors_tensor.shape[-1])
    
    print(f"descriptors_tensor shape before norm: {descriptors_tensor.shape}")
    norm_descs = torch.nn.functional.normalize(descriptors_tensor, p=2, dim=1)  # (m, 256)
    print(f"norm_descs shape: {norm_descs.shape}")
    similarity_matrix = norm_descs @ norm_descs.t()  # (m, m) - cosine similarity
    print(f"similarity_matrix shape: {similarity_matrix.shape}")
    
    # 为每张图找到最相似的K张或基于阈值找相似图
    match_pairs = []
    
    if threshold is not None:
        # 基于阈值的匹配
        print(f"Using similarity threshold: {threshold}")
        for i in range(len(valid_images)):
            similarities = similarity_matrix[i]  # (m,) - 第i张图与所有图的相似度
            
            # 找出所有相似度大于阈值的索引（排除自己，自己的相似度为1）
            similar_mask = similarities > threshold
            similar_indices = torch.where(similar_mask)[0]
            print(f"Image {i} has {len(similar_indices)} similar images above threshold {threshold}")
            
            # 如果大于阈值的匹配太多，只取前30个最相似的
            max_similar_num = 30
            if len(similar_indices) > max_similar_num:
                # 获取这些索引对应的相似度值
                similar_sims = similarities[similar_indices]
                # 排序并取前30个最相似的
                top_vals, top_local_idx = torch.topk(similar_sims, min(max_similar_num, len(similar_sims)))
                similar_indices = similar_indices[top_local_idx]
                print(f"  Keeping top-{max_similar_num} similar images (limited from {len(similar_sims)})")

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
                    match_pairs.append((i_orig, img_name0, j_orig, img_name1))
    else:
        # Top-K 匹配
        for i in range(len(valid_images)):
            similarities = similarity_matrix[i]  # (m,) - 第i张图与所有图的相似度
            # print(f"Image {i} similarities shape: {similarities.shape}")
            
            # 使用 torch.topk 获取top-K相似的索引（排除自己）
            topk_vals, topk_indices = torch.topk(similarities, min(k+1, len(valid_images)))
            # print(f"topk_indices shape: {topk_indices.shape}")
            print(f"topk_vals: {topk_vals}")
            
            # 跳过第一个（自己），取后面的k个
            for idx_in_topk in range(1, min(k+1, len(topk_indices))):
                j_local = int(topk_indices[idx_in_topk].item())  # 转换为Python整数
                
                img_name0 = valid_images[i]
                img_name1 = valid_images[j_local]
                
                # 找到原始image_list中的索引
                i_orig = image_list.index(img_name0)
                j_orig = image_list.index(img_name1)
                
                if i_orig < j_orig:  # 避免重复
                    match_pairs.append((i_orig, img_name0, j_orig, img_name1))
    
    if logger:
        if threshold is not None:
            logger.info(f"Found {len(match_pairs)} image pairs using similarity threshold ({threshold})")
        else:
            logger.info(f"Found {len(match_pairs)} image pairs using parallel similarity search (top-{k})")
    
    return match_pairs


def find_similar_images_quick(image_features, k=10, logger=None):
    """
    快速启发式方法：基于特征点分布做粗筛 + 描述符做精筛。
    适合大规模数据集，速度快。
    
    Args:
        image_features: Dict of image features
        k: 每张图找到的最相似图像数
        logger: Optional logger instance
    
    Returns:
        List of (i, img_name0, j, img_name1) tuples representing match pairs
    """
    image_list = list(image_features.keys())
    
    # Step 1: 基于特征点统计的粗筛
    feature_stats = {}
    for img_name in image_list:
        if image_features[img_name] is not None:
            kpts = image_features[img_name]['keypoints']
            if len(kpts) > 0:
                feature_stats[img_name] = {
                    'num_kpts': len(kpts),
                    'center': kpts.mean(dim=0) if torch.is_tensor(kpts) else kpts.mean(axis=0),
                    'spread': kpts.std(dim=0) if torch.is_tensor(kpts) else kpts.std(axis=0)
                }
    
    # Step 2: 找候选（数量是k的2-3倍）
    match_pairs = []
    candidate_k = min(k * 2, len(image_list) - 1)
    
    for i, img_name0 in enumerate(image_list):
        stats0 = feature_stats.get(img_name0)
        if stats0 is None:
            continue
        
        # 基于特征点数量的粗相似度
        candidates = []
        for j, img_name1 in enumerate(image_list):
            if i >= j:
                continue
            
            stats1 = feature_stats.get(img_name1)
            if stats1 is None:
                continue
            
            # 特征点数量的相似度（0-1）
            kpt_similarity = 1 - abs(stats0['num_kpts'] - stats1['num_kpts']) / (
                max(stats0['num_kpts'], stats1['num_kpts']) + 1e-8
            )
            
            candidates.append((j, img_name1, kpt_similarity))
        
        # 按初步相似度排序，选择top候选
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # 对候选进行精细相似度计算
        feats0 = image_features[img_name0]
        if feats0 is not None:
            desc0 = feats0['descriptors'].mean(dim=1)
            
            refined_candidates = []
            for j, img_name1, _ in candidates[:candidate_k]:
                feats1 = image_features[img_name1]
                if feats1 is not None:
                    desc1 = feats1['descriptors'].mean(dim=1)
                    
                    similarity = torch.nn.functional.cosine_similarity(desc0, desc1, dim=-1).item()
                    refined_candidates.append((j, img_name1, similarity))
            
            # 选择最相似的K个
            refined_candidates.sort(key=lambda x: x[2], reverse=True)
            for j, img_name1, _ in refined_candidates[:k]:
                match_pairs.append((i, img_name0, j, img_name1))
    
    if logger:
        logger.info(f"Found {len(match_pairs)} image pairs using quick heuristic filtering")
    
    return match_pairs


def extract_features_with_superpoint(
    database_path: str,
    images_path: str,
    max_num_keypoints: int = 2048,
    logger: logging.Logger = None
) -> tuple:
    """
    Extract features using SuperPoint and write to COLMAP database.
    
    Args:
        database_path: Path to COLMAP database
        images_path: Path to images directory
        max_num_keypoints: Maximum number of keypoints per image
        logger: Optional logger instance
    
    Returns:
        (image_features, image_id_map, feature_extraction_time)
        - image_features: Dict of image name -> feature tensor
        - image_id_map: Dict of image name -> image_id in database
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Initialize SuperPoint model
    logger.info("Initializing SuperPoint model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    weights_path = "checkpoints/superpoint_v1.pth"
    extractor = SuperPoint(weights_path=weights_path, max_num_keypoints=max_num_keypoints).eval().to(device)
    
    # Get list of images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(images_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    if not image_files:
        logger.error(f"No images found in {images_path}")
        return {}, {}, 0
    
    logger.info(f"Found {len(image_files)} images")
    
    # Connect to COLMAP database
    try:
        db = COLMAPDatabase.connect(database_path)
        cursor = db.cursor()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return {}, {}, 0
    
    # Extract features for all images
    logger.info("Extracting features with SuperPoint...")
    feature_start = time.time()
    
    image_features = {}
    image_id_map = {}
    total_keypoints = 0
    
    # Buffer for batch write operations
    keypoints_to_write = []
    extraction_log = []
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_path, img_file)
        try:
            # Load image
            img_tensor = load_image(img_path).to(device)
            
            # Extract features
            with torch.no_grad():
                feats = extractor.extract(img_tensor)
            
            # Get image ID from database
            cursor.execute("SELECT image_id FROM images WHERE name = ?", (img_file,))
            result = cursor.fetchone()
            
            if result is None:
                logger.warning(f"Image {img_file} not found in database, skipping feature write")
                image_features[img_file] = feats
                image_id_map[img_file] = None
                continue
            
            image_id = result[0]
            image_features[img_file] = feats
            image_id_map[img_file] = image_id
            
            # Extract keypoints and descriptors (remove batch dimension)
            keypoints_batch = feats['keypoints']
            descriptors_batch = feats['descriptors']
            
            # Remove batch dimension if present
            if keypoints_batch.dim() == 3 and keypoints_batch.shape[0] == 1:
                keypoints = keypoints_batch[0]
                descriptors = descriptors_batch[0]
            else:
                keypoints = keypoints_batch
                descriptors = descriptors_batch
            
            # Convert to numpy
            keypoints = keypoints.cpu().numpy()
            descriptors = descriptors.cpu().numpy()
            
            # Descriptors from SuperPoint are Float32 in range [0, 1], convert to uint8
            descriptors = (descriptors * 255.0).astype(np.uint8)
            
            num_kpts = keypoints.shape[0]
            total_keypoints += num_kpts
            
            # Buffer keypoints for batch write
            keypoints_to_write.append((image_id, keypoints, descriptors))
            extraction_log.append((idx, img_file, num_kpts, image_id))
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {img_file}: {e}")
            image_features[img_file] = None
            continue
    
    # Batch write all keypoints and descriptors to database
    logger.info(f"Writing {len(keypoints_to_write)} images' keypoints to database...")
    for image_id, keypoints, descriptors in keypoints_to_write:
        write_keypoints_to_database(db, image_id, keypoints, descriptors, logger)
    
    # Log extraction results
    for idx, img_file, num_kpts, image_id in extraction_log:
        logger.info(f"  [{idx+1}/{len(image_files)}] {img_file}: {num_kpts} keypoints (image_id={image_id})")
    
    feature_extraction_time = time.time() - feature_start
    successful_extractions = sum(1 for f in image_features.values() if f is not None)
    logger.info(f"Feature extraction completed in {feature_extraction_time:.2f}s")
    logger.info(f"  Extracted features from {successful_extractions}/{len(image_files)} images")
    logger.info(f"  Total keypoints extracted: {total_keypoints}")
    if successful_extractions > 0:
        logger.info(f"  Average keypoints per image: {total_keypoints/successful_extractions:.1f}")
    
    db.close()
    
    return image_features, image_id_map, feature_extraction_time


def match_features_with_lightglue(
    database_path: str,
    image_features: dict,
    image_id_map: dict,
    match_list_path: str = None,
    match_strategy: str = "nearest_k",
    max_matches_per_image: int = 15,
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
    match_strategy_threshold = similarity_threshold  # 使匹配策略函数中可访问
    if logger is None:
        logger = logging.getLogger()
    
    # Initialize LightGlue matcher
    logger.info("Initializing LightGlue matcher...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = "checkpoints/superpoint_lightglue_v0-1_arxiv.pth"
    matcher = LightGlue(path_or_url=weight_path).eval().to(device)
    
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
    
    # 根据策略选择匹配对
    if match_strategy == "exhaustive":
        match_pairs = []
        for i, img_name0 in enumerate(image_list):
            for j, img_name1 in enumerate(image_list):
                if i < j:
                    match_pairs.append((i, img_name0, j, img_name1))
        logger.info(f"Exhaustive matching: {len(match_pairs)} image pairs will be matched")
    
    elif match_strategy == "nearest_k":
        logger.info(f"Finding top-{max_matches_per_image} similar images for each (parallel)...")
        similarity_start = time.time()
        match_pairs = find_similar_images_parallel(image_features, k=max_matches_per_image, threshold=None, logger=logger)
        logger.info(f"Similarity computation took {time.time() - similarity_start:.2f}s ({len(match_pairs)} pairs)")
    
    elif match_strategy == "threshold":
        logger.info(f"Using similarity threshold matching (threshold={match_strategy_threshold})...")
        similarity_start = time.time()
        match_pairs = find_similar_images_parallel(image_features, k=None, threshold=match_strategy_threshold, logger=logger)
        logger.info(f"Similarity computation took {time.time() - similarity_start:.2f}s ({len(match_pairs)} pairs)")
    
    elif match_strategy == "quick":
        logger.info(f"Using quick heuristic filtering (top-{max_matches_per_image})...")
        similarity_start = time.time()
        match_pairs = find_similar_images_quick(image_features, k=max_matches_per_image, logger=logger)
        logger.info(f"Quick filtering took {time.time() - similarity_start:.2f}s ({len(match_pairs)} pairs)")
    
    else:
        logger.warning(f"Unknown match_strategy '{match_strategy}', falling back to exhaustive")
        match_pairs = []
        for i, img_name0 in enumerate(image_list):
            for j, img_name1 in enumerate(image_list):
                if i < j:
                    match_pairs.append((i, img_name0, j, img_name1))
    
    # Buffer for batch write operations
    matches_to_write = []
    match_list_pairs = []
    
    # Perform matching
    logger.info(f"Performing LightGlue matching on {len(match_pairs)} pairs...")
    match_progress_interval = max(1, len(match_pairs) // 100)
    
    for pair_idx, (i, img_name0, j, img_name1) in enumerate(match_pairs):
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
            
        except Exception as e:
            logger.warning(f"Failed to match {img_name0} and {img_name1}: {e}")
            continue
    
    # Batch write all matches to database
    logger.info(f"Writing {total_match_pairs} match pairs to database...")
    for image_id0, image_id1, matches in matches_to_write:
        write_matches_to_database(db, image_id0, image_id1, matches, logger)
    
    # Batch write match list to file if path provided
    if match_list_path:
        try:
            os.makedirs(os.path.dirname(match_list_path), exist_ok=True)
            with open(match_list_path, 'w') as match_file:
                for img_name0, img_name1 in match_list_pairs:
                    match_file.write(f"{img_name0} {img_name1}\n")
            logger.info(f"Match list written to {match_list_path}")
        except Exception as e:
            logger.warning(f"Could not write match_list_path file: {e}")
    
    feature_matching_time = time.time() - matching_start
    logger.info(f"Feature matching by lightglue completed in {feature_matching_time:.2f}s")
    logger.info(f"  Matched {total_match_pairs} image pairs")
    logger.info(f"  Total matches found: {total_matches}")
    if total_match_pairs > 0:
        logger.info(f"  Average matches per pair: {total_matches/total_match_pairs:.1f}")
    logger.info("SuperPoint + LightGlue features and matches written to database")
    
    db.close()
    
    return feature_matching_time


# --------------------------
# Feature extraction & matching
# --------------------------
logger.info("=== Starting Structure-from-Motion Pipeline ===")
logger.info(f"Source path: {args.source_path}")
logger.info(f"Output path: {output_path}")
distorted_sparse_path = os.path.join(output_path, "distorted/sparse")
os.makedirs(distorted_sparse_path, exist_ok=True)

database_path = os.path.join(output_path, "distorted/database.db")
match_list_path = os.path.join(distorted_sparse_path, "image_pairs_to_match.txt")
images_path = os.path.join(args.source_path, "input")

# --- Feature extraction ---
if args.SuperpointLightglue:
    logger.info("Using SuperPoint + LightGlue for feature extraction and matching...")
    logger.info(f"  Match strategy: {args.match_strategy}")
    if args.match_strategy in ["nearest_k", "quick"]:
        logger.info(f"  Max matches per image: {args.max_matches_per_image}")
    
    # First, initialize COLMAP database with images using COLMAP's feature_extractor
    logger.info("Initializing COLMAP database with image metadata...")
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_path,
        "--ImageReader.single_camera_per_image", str(args.single_image),
        "--ImageReader.single_camera_per_fold", str(args.single_fold),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.camera,
        "--FeatureExtraction.use_gpu", str(use_gpu),
        "--log_level", str(log_level),
    ]
    run_subprocess(feat_extraction_cmd, logger)
    
    # Now run SuperPoint feature extraction
    splg_start = time.time()
    image_features, image_id_map, feat_ext_time = extract_features_with_superpoint(
        database_path, 
        images_path, 
        max_num_keypoints=args.max_feature_num,
        logger=logger
    )
    feature_extraction_time = feat_ext_time
    
    # Run LightGlue feature matching
    feat_match_time = match_features_with_lightglue(
        database_path,
        image_features,
        image_id_map,
        match_list_path=match_list_path,
        match_strategy=args.match_strategy,
        max_matches_per_image=args.max_matches_per_image,
        similarity_threshold=args.similarity_threshold,
        logger=logger
    )
    feature_matching_time = feat_match_time
    splg_time = time.time() - splg_start
    
    # --- Import matches using COLMAP matches_importer ---
    # This is the recommended way to import pre-computed matches into COLMAP database
    # Following the same approach as super_colmap.py
    logger.info("Importing matches into COLMAP database using matches_importer...")
    matches_importer_cmd = [
        colmap_command, "matches_importer",
        "--database_path", database_path,
        "--match_list_path", match_list_path,
        "--match_type", "pairs",
        "--TwoViewGeometry.min_num_inliers", str(min_num_inliers),
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio),
        "--TwoViewGeometry.multiple_models", "0",
        "--TwoViewGeometry.compute_relative_pose", "0",
        "--TwoViewGeometry.detect_watermark", "0",
        "--TwoViewGeometry.multiple_ignore_watermark", "0",
        "--TwoViewGeometry.watermark_detection_max_error", "4",
        "--TwoViewGeometry.filter_stationary_matches", "0",
        "--TwoViewGeometry.stationary_matches_max_error", "4",
        "--TwoViewGeometry.max_error", "4",
        "--TwoViewGeometry.confidence", "0.9999",
        "--TwoViewGeometry.max_num_trials", "10000"
    ]
    run_subprocess(matches_importer_cmd, logger)
    logger.info("Matches imported successfully")
else:
    # Use COLMAP's built-in feature extractor
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_path,
        "--ImageReader.single_camera_per_image", str(args.single_image),
        "--ImageReader.single_camera_per_fold", str(args.single_fold),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.camera,
        "--SiftExtraction.max_num_features", str(args.max_feature_num),
        # "--SiftExtraction.peak_threshold", "0.00667", # 0.00667
        # "--SiftExtraction.max_num_orientations", "2", # 2
        "--FeatureExtraction.use_gpu", str(use_gpu),
        "--log_level", str(log_level),
    ]
    logger.info("Starting feature extraction with COLMAP SIFT...")
    t0 = time.time()
    run_subprocess(feat_extraction_cmd, logger)
    feature_extraction_time = time.time() - t0
    logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

    # --- Feature matching ---
    logger.info("Starting feature matching...")
    t1 = time.time()
    if args.alg == "default":
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--log_level", str(log_level),
        ]
    else:
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--ExhaustiveMatching.block_size", "200", # 50
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.guided_matching", "0", # 0
            "--SiftMatching.max_ratio", "0.8", # 0.8
            "--SiftMatching.max_distance", "0.7", # 0.7
            "--TwoViewGeometry.min_num_inliers", str(min_num_inliers), # 15
            "--TwoViewGeometry.max_error", "4", # 4
            "--TwoViewGeometry.confidence", "0.9999", # 0.999
            "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
            "--TwoViewGeometry.detect_watermark", "0", # 1
            "--TwoViewGeometry.filter_stationary_matches", "0", # 0
            "--TwoViewGeometry.compute_relative_pose", "0", # 0
            "--log_level", str(log_level), # 0
        ]

    run_subprocess(feat_matching_cmd, logger)
    feature_matching_time = time.time() - t1
    logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

# --------------------------
# Mapper / Bundle Adjustment
# --------------------------
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper...")
t2 = time.time()
if args.alg == "acc":
    mapper_cmd = [
        colmap_command, "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.num_threads", "-1",
        "--Mapper.min_num_matches", "25", # 15
        "--Mapper.init_num_trials", "200", # 200
        "--Mapper.init_min_num_inliers", "100", # 100
        "--Mapper.init_max_error", "4", # 4
        "--Mapper.init_min_tri_angle", "16", # 16
        "--Mapper.ba_local_min_tri_angle", "6", # 6
        "--Mapper.ba_local_num_images", "6", # 6
        "--Mapper.ba_local_max_num_iterations", "12", # 25
        "--Mapper.ba_local_max_refinements", "2", # 2
        "--Mapper.ba_local_max_refinement_change", "0.001", # 0.001
        "--Mapper.ba_global_frames_ratio", "2.0", # 1.1
        "--Mapper.ba_global_points_ratio", "2.0", # 1.1
        "--Mapper.ba_global_frames_freq", "5000", # 5000
        "--Mapper.ba_global_points_freq", "250000", # 250000
        "--Mapper.ba_global_max_num_iterations", "20", # 50
        "--Mapper.ba_global_max_refinements", "2", # 5
        "--Mapper.ba_global_max_refinement_change", "0.001", # 0.0005
        "--Mapper.ba_refine_focal_length", "1", # 1
        "--Mapper.ba_refine_principal_point", "0", # 0
        "--Mapper.ba_refine_extra_params", "1", # 1
        "--Mapper.abs_pose_max_error", "12", # 12
        "--Mapper.abs_pose_min_num_inliers", str(min_num_inliers), # 30
        "--Mapper.abs_pose_min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--Mapper.max_extra_param", "1", # 1
        "--Mapper.tri_min_angle", "1.5", # 1.5
        "--Mapper.tri_create_max_angle_error", "2", # 2
        "--Mapper.tri_merge_max_reproj_error", "4", # 4
        "--Mapper.filter_max_reproj_error", "4", # 4
        "--Mapper.max_reg_trials", "3", # 3
        "--log_level", str(log_level),
    ]
elif args.alg == "hierarchical":
    mapper_cmd = [
        colmap_command, "hierarchical_mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--log_level", str(log_level),
        "--num_workers", "-1",
        "--image_overlap", "5", # 50
        "--leaf_max_num_images", "50", # 500
        "--Mapper.min_num_matches", "15",
        "--Mapper.ignore_watermarks", "0",
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "50",
        "--Mapper.max_model_overlap", "20",
        "--Mapper.min_model_size", "10",
        "--Mapper.init_num_trials", "200",
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", "-1",
        "--Mapper.random_seed", "-1",
        "--Mapper.min_focal_length_ratio", "0.1",
        "--Mapper.max_focal_length_ratio", "10",
        "--Mapper.max_extra_param", "1",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_refine_sensor_from_rig", "1",
        "--Mapper.ba_local_function_tolerance", "0",
        "--Mapper.ba_local_max_num_iterations", "25", #25
        "--Mapper.ba_global_frames_ratio", "1.1",
        "--Mapper.ba_global_points_ratio", "1.1",
        "--Mapper.ba_global_frames_freq", "500",
        "--Mapper.ba_global_points_freq", "250000",
        "--Mapper.ba_global_function_tolerance", "0",
        "--Mapper.ba_global_max_num_iterations", "50", # 50
        "--Mapper.ba_global_max_refinements", "5", # 5
        "--Mapper.ba_global_max_refinement_change", "0.0005",
        "--Mapper.ba_local_max_refinements", "2", # 2
        "--Mapper.ba_local_max_refinement_change", "0.001",
        "--Mapper.ba_use_gpu", "0",
        "--Mapper.ba_gpu_index", "-1",
        "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        "--Mapper.snapshot_path", "",
        "--Mapper.snapshot_frames_freq", "0",
        "--Mapper.fix_existing_frames", "0",
        "--Mapper.init_min_num_inliers", "100",
        "--Mapper.init_max_error", "4",
        "--Mapper.init_max_forward_motion", "0.95",
        "--Mapper.init_min_tri_angle", "16",
        "--Mapper.init_max_reg_trials", "2",
        "--Mapper.abs_pose_max_error", "12",
        "--Mapper.abs_pose_min_num_inliers", str(min_num_inliers),
        "--Mapper.abs_pose_min_inlier_ratio", str(min_inlier_ratio),
        "--Mapper.filter_max_reproj_error", "4",
        "--Mapper.filter_min_tri_angle", "1.5",
        "--Mapper.max_reg_trials", "3",
        "--Mapper.ba_local_num_images", "6",
        "--Mapper.ba_local_min_tri_angle", "6",
        "--Mapper.ba_global_ignore_redundant_points3D", "0",
        "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", "0.05",
        "--Mapper.image_list_path", "",
        "--Mapper.constant_rig_list_path", "",
        "--Mapper.constant_camera_list_path", "",
        "--Mapper.max_runtime_seconds", "-1",
        "--Mapper.tri_max_transitivity", "1",
        "--Mapper.tri_create_max_angle_error", "2",
        "--Mapper.tri_continue_max_angle_error", "2",
        "--Mapper.tri_merge_max_reproj_error", "4",
        "--Mapper.tri_complete_max_reproj_error", "4",
        "--Mapper.tri_complete_max_transitivity", "5",
        "--Mapper.tri_re_max_angle_error", "5",
        "--Mapper.tri_re_min_ratio", "0.2",
        "--Mapper.tri_re_max_trials", "1",
        "--Mapper.tri_min_angle", "1.5",
        "--Mapper.tri_ignore_two_view_tracks", "1"
    ]

elif args.alg == "hierarchical_acc":
    mapper_cmd = [
        colmap_command, "hierarchical_mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--log_level", str(log_level),
        "--num_workers", "-1",
        "--image_overlap", "5", # 50
        "--leaf_max_num_images", "50", # 500
        "--Mapper.min_num_matches", "15",
        "--Mapper.ignore_watermarks", "0",
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "50",
        "--Mapper.max_model_overlap", "20",
        "--Mapper.min_model_size", "10",
        "--Mapper.init_num_trials", "200",
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", "-1",
        "--Mapper.random_seed", "-1",
        "--Mapper.min_focal_length_ratio", "0.1",
        "--Mapper.max_focal_length_ratio", "10",
        "--Mapper.max_extra_param", "1",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_refine_sensor_from_rig", "1",
        "--Mapper.ba_local_function_tolerance", "0",
        "--Mapper.ba_local_max_num_iterations", "12", #25
        "--Mapper.ba_global_frames_ratio", "2.0",
        "--Mapper.ba_global_points_ratio", "2.0",
        "--Mapper.ba_global_frames_freq", "500",
        "--Mapper.ba_global_points_freq", "250000",
        "--Mapper.ba_global_function_tolerance", "0",
        "--Mapper.ba_global_max_num_iterations", "20", # 50
        "--Mapper.ba_global_max_refinements", "2", # 5
        "--Mapper.ba_global_max_refinement_change", "0.0005",
        "--Mapper.ba_local_max_refinements", "2", # 2
        "--Mapper.ba_local_max_refinement_change", "0.001",
        "--Mapper.ba_use_gpu", "0",
        "--Mapper.ba_gpu_index", "-1",
        "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        "--Mapper.snapshot_path", "",
        "--Mapper.snapshot_frames_freq", "0",
        "--Mapper.fix_existing_frames", "0",
        "--Mapper.init_min_num_inliers", "100",
        "--Mapper.init_max_error", "4",
        "--Mapper.init_max_forward_motion", "0.95",
        "--Mapper.init_min_tri_angle", "16",
        "--Mapper.init_max_reg_trials", "2",
        "--Mapper.abs_pose_max_error", "12",
        "--Mapper.abs_pose_min_num_inliers", str(min_num_inliers),
        "--Mapper.abs_pose_min_inlier_ratio", str(min_inlier_ratio),
        "--Mapper.filter_max_reproj_error", "4",
        "--Mapper.filter_min_tri_angle", "1.5",
        "--Mapper.max_reg_trials", "3",
        "--Mapper.ba_local_num_images", "6",
        "--Mapper.ba_local_min_tri_angle", "6",
        "--Mapper.ba_global_ignore_redundant_points3D", "0",
        "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", "0.05",
        "--Mapper.image_list_path", "",
        "--Mapper.constant_rig_list_path", "",
        "--Mapper.constant_camera_list_path", "",
        "--Mapper.max_runtime_seconds", "-1",
        "--Mapper.tri_max_transitivity", "1",
        "--Mapper.tri_create_max_angle_error", "2",
        "--Mapper.tri_continue_max_angle_error", "2",
        "--Mapper.tri_merge_max_reproj_error", "4",
        "--Mapper.tri_complete_max_reproj_error", "4",
        "--Mapper.tri_complete_max_transitivity", "5",
        "--Mapper.tri_re_max_angle_error", "5",
        "--Mapper.tri_re_min_ratio", "0.2",
        "--Mapper.tri_re_max_trials", "1",
        "--Mapper.tri_min_angle", "1.5",
        "--Mapper.tri_ignore_two_view_tracks", "1"
    ]


elif args.alg == "glomap":
    mapper_cmd = [
        glomap_command, "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--ba_iteration_num", "3",
        "--retriangulation_iteration_num", "1",
        "--skip_preprocessing", "0",
        "--skip_view_graph_calibration", "0",
        "--skip_relative_pose_estimation", "0",
        "--skip_rotation_averaging", "0",
        "--skip_global_positioning", "0",
        "--skip_bundle_adjustment", "0",
        "--skip_retriangulation", "0",
        "--skip_pruning", "0", # 1
        "--ViewGraphCalib.thres_lower_ratio", "0.1",
        "--ViewGraphCalib.thres_higher_ratio", "10",
        "--ViewGraphCalib.thres_two_view_error", "2",
        "--RelPoseEstimation.max_epipolar_error", "1",
        "--TrackEstablishment.min_num_tracks_per_view", "-1",
        "--TrackEstablishment.min_num_view_per_track", "3",
        "--TrackEstablishment.max_num_view_per_track", "100",
        "--TrackEstablishment.max_num_tracks", "10000000",
        "--GlobalPositioning.use_gpu", "1",
        "--GlobalPositioning.gpu_index", "-1",
        "--GlobalPositioning.optimize_positions", "1",
        "--GlobalPositioning.optimize_points", "1",
        "--GlobalPositioning.optimize_scales", "1",
        "--GlobalPositioning.thres_loss_function", "0.1", # 0.1
        "--GlobalPositioning.max_num_iterations", "100",
        "--BundleAdjustment.use_gpu", "1",
        "--BundleAdjustment.gpu_index", "-1",
        "--BundleAdjustment.optimize_rig_poses", "0",
        "--BundleAdjustment.optimize_rotations", "1",
        "--BundleAdjustment.optimize_translation", "1",
        "--BundleAdjustment.optimize_intrinsics", "1",
        "--BundleAdjustment.optimize_principal_point", "0",
        "--BundleAdjustment.optimize_points", "1",
        "--BundleAdjustment.thres_loss_function", "1",
        "--BundleAdjustment.max_num_iterations", "200",
        "--Triangulation.complete_max_reproj_error", "15",
        "--Triangulation.merge_max_reproj_error", "15",
        "--Triangulation.min_angle", "2", # 1
        "--Triangulation.min_num_matches", "15", # 15
        "--Thresholds.max_angle_error", "1",
        "--Thresholds.max_reprojection_error", "0.01",
        "--Thresholds.min_triangulation_angle", "3", # 1
        "--Thresholds.max_epipolar_error_E", "1",
        "--Thresholds.max_epipolar_error_F", "4",
        "--Thresholds.max_epipolar_error_H", "4",
        "--Thresholds.min_inlier_num", str(min_num_inliers), # 30
        "--Thresholds.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--Thresholds.max_rotation_error", "10"
    ]
else:
    mapper_cmd = [
        colmap_command, "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--log_level", str(log_level),
    ]
run_subprocess(mapper_cmd, logger)
mapper_time = time.time() - t2
logger.info(f"Mapper done in {mapper_time:.2f} s")

# --------------------------
# Image undistortion
# --------------------------
statr_undistort = time.time()
largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
img_undist_cmd = [
    colmap_command, "image_undistorter",
    "--image_path", images_path,
    "--input_path", largest_sparse_folder,
    "--output_path", output_path,
    "--output_type", "COLMAP"
]
run_subprocess(img_undist_cmd, logger)
undistort_time = time.time() - statr_undistort
logger.info(f"Image undistortion done, used time: {undistort_time:.2f} s.")

# --------------------------
# Organize sparse output to sparse/0
# --------------------------
sparse_output_path = os.path.join(output_path, "sparse")
os.makedirs(os.path.join(sparse_output_path, "0"), exist_ok=True)

logger.info("Organizing sparse output files into sparse/0 ...")
for item in os.listdir(sparse_output_path):
    src_path = os.path.join(sparse_output_path, item)
    dst_path = os.path.join(sparse_output_path, "0", item)
    if os.path.isdir(src_path):
        if item == "0":
            continue
        for f in os.listdir(src_path):
            f_src = os.path.join(src_path, f)
            f_dst = os.path.join(sparse_output_path, "0", f)
            shutil.move(f_src, f_dst)
        os.rmdir(src_path)
    elif os.path.isfile(src_path):
        shutil.move(src_path, dst_path)

img_num = count_images_in_dir(os.path.join(output_path, "images"))
input_img_num = count_images_in_dir_recursive(images_path)
logger.info("Sparse output successfully organized into sparse/0.")

# --------------------------
# Timing summary
# --------------------------
total_time = time.time() - start_time
sparse_reconstruction_time = feature_extraction_time + feature_matching_time + mapper_time
pair_match_time = feature_matching_time / (input_img_num * (input_img_num - 1) / 2) if input_img_num > 1 else 0

logger.info("Done. Timing statistics:")
logger.info(f"  Feature extraction: {int(feature_extraction_time // 60)} min {feature_extraction_time % 60:.2f} s")
logger.info(f"  Feature matching: {int(feature_matching_time // 60)} min {feature_matching_time % 60:.2f} s")
logger.info(f"  Average time per image pair: {pair_match_time * 1000:.2f} ms")
logger.info(f"  Mapper: {int(mapper_time // 60)} min {mapper_time % 60:.2f} s")
logger.info(f"  Sparse reconstruction: {int(sparse_reconstruction_time // 60)} min {sparse_reconstruction_time % 60:.2f} s")
logger.info(f"  Image undistortion: {int(undistort_time // 60)} min {undistort_time % 60:.2f} s")
logger.info(f"  Total time: {int(total_time // 60)} min {total_time % 60:.2f} s")
logger.info(f"  Number of images registered: {img_num} / {input_img_num}")