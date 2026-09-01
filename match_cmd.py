import os
import logging

def get_sequential_match_list(
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
                             max_distance=0.7,
                             max_ratio=0.5,                                
                             two_view_geometry_max_error: float = 4.0,                            
                             sift_lightglue_match_path: str = "",
                             bruteforce_match_path: str = "",
                             aliked_lightglue_match_path: str = "",
                             loma_match_min_score: float = 0.1,
                             loma_match_use_bf16: int = 0,
                             b_model_path: str = "",
                             b_model_path_bf16: str = "",
                             b128_model_path: str = "",
                             b128_model_path_bf16: str = "",
                             r_model_path: str = "",
                             r_model_path_bf16: str = "",
                             l_model_path: str = "",
                             l_model_path_bf16: str = "",
                             g_model_path: str = "",
                             g_model_path_bf16: str = ""
):
    matches_importer_cmd = [
        colmap_command, "matches_importer",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--match_list_path", matched_images_pairs_path,
        "--match_type", "pairs", # {'pairs', 'raw', 'inliers'}               
        "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT, ALIKED_LIGHTGLUE, ALIKED_N32
        "--FeatureMatching.num_threads", "-1",
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.guided_matching", "0",
        "--FeatureMatching.skip_geometric_verification", "0",
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", str(max_ratio), # 0.8
        "--SiftMatching.max_distance", str(max_distance), # 0.7
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
        "--LomaMatching.min_score", str(loma_match_min_score),
        "--LomaMatching.use_bf16", str(loma_match_use_bf16),
        "--LomaMatching.b_model_path", b_model_path,
        "--LomaMatching.b_model_path_bf16", b_model_path_bf16,
        "--LomaMatching.b128_model_path", b128_model_path,
        "--LomaMatching.b128_model_path_bf16", b128_model_path_bf16,
        "--LomaMatching.r_model_path", r_model_path,
        "--LomaMatching.r_model_path_bf16", r_model_path_bf16,
        "--LomaMatching.l_model_path", l_model_path,
        "--LomaMatching.l_model_path_bf16", l_model_path_bf16,
        "--LomaMatching.g_model_path", g_model_path,
        "--LomaMatching.g_model_path_bf16", g_model_path_bf16,
        "--LomaMatching.brute_force_min_cossim", "0.85",
        "--LomaMatching.brute_force_max_ratio", "1",
        "--LomaMatching.brute_force_cross_check", "1",
        "--LomaMatching.brute_force_model_path", bruteforce_match_path,          
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
                             loma_match_min_score: float = 0.1,
                             loma_match_use_bf16: int = 0,
                             b_model_path: str = "",
                             b_model_path_bf16: str = "",
                             b128_model_path: str = "",
                             b128_model_path_bf16: str = "",
                             r_model_path: str = "",
                             r_model_path_bf16: str = "",
                             l_model_path: str = "",
                             l_model_path_bf16: str = "",
                             g_model_path: str = "",
                             g_model_path_bf16: str = "",                             
                             sift_match_max_distance: float=0.7,
                             sift_match_max_ratio: float=0.8,
                             two_view_geometry_max_error: float=4.0):

    if feature_match_type.lower().startswith("sift"):
        guided_matching = "1"
    else:
        guided_matching = "0"

    exhaustive_matcher_cmd = [
        colmap_command, "exhaustive_matcher",
        "--log_level", str(log_level), # 0
        "--database_path", database_path,
        "--ExhaustiveMatching.block_size", "200", # 50 
        "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT_BRUTEFORCE, ALIKED_LIGHTGLUE, ALIKED_N32
        # "--FeatureMatching.num_threads", str(args.num_threads),
        "--FeatureMatching.use_gpu", str(use_gpu),
        # "--FeatureMatching.gpu_index", str(args.gpu_index),
        "--FeatureMatching.guided_matching", guided_matching,
        # "--FeatureMatching.skip_geometric_verification", str(args.skip_geometric_verification),
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", str(sift_match_max_ratio), # 0.8
        "--SiftMatching.max_distance", str(sift_match_max_distance), # 0.7
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
        "--LomaMatching.min_score", str(loma_match_min_score),
        "--LomaMatching.use_bf16", str(loma_match_use_bf16),
        "--LomaMatching.b_model_path", b_model_path,
        "--LomaMatching.b_model_path_bf16", b_model_path_bf16,
        "--LomaMatching.b128_model_path", b128_model_path,
        "--LomaMatching.b128_model_path_bf16", b128_model_path_bf16,
        "--LomaMatching.r_model_path", r_model_path,
        "--LomaMatching.r_model_path_bf16", r_model_path_bf16,
        "--LomaMatching.l_model_path", l_model_path,
        "--LomaMatching.l_model_path_bf16", l_model_path_bf16,
        "--LomaMatching.g_model_path", g_model_path,
        "--LomaMatching.g_model_path_bf16", g_model_path_bf16,
        "--LomaMatching.brute_force_min_cossim", "0.85",
        "--LomaMatching.brute_force_max_ratio", "1",
        "--LomaMatching.brute_force_cross_check", "1",
        "--LomaMatching.brute_force_model_path", bruteforce_match_path,        
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
                             vocab_path: str,                             
                             sift_lightglue_match_path: str,
                             bruteforce_match_path: str,
                             aliked_lightglue_match_path: str, 
                             loma_match_min_score: float = 0.1,
                             loma_match_use_bf16: int = 0,
                             b_model_path: str = "",
                             b_model_path_bf16: str = "",
                             b128_model_path: str = "",
                             b128_model_path_bf16: str = "",
                             r_model_path: str = "",
                             r_model_path_bf16: str = "",
                             l_model_path: str = "",
                             l_model_path_bf16: str = "",
                             g_model_path: str = "",
                             g_model_path_bf16: str = "",  
                             vocab_feature_num: int = 0,                             
                             max_feature_num: int = 2048,
                             max_matches_per_image: int = 150,
                             min_matches_per_image: int = 0,
                             min_num_inliers: int = 15,
                             min_inlier_ratio: float = 0.25,
                             sift_match_max_distance: float = 0.7,
                             sift_match_max_ratio: float = 0.5,                             
                             two_view_geometry_max_error: float = 4.0
                             ):

    if feature_match_type.lower().startswith("sift"):
        guided_matching = "1"
    else:
        guided_matching = "0"
    vocab_tree_matcher_cmd = [
        colmap_command, "vocab_tree_matcher",
        "--log_level", str(log_level),
        "--database_path", database_path,        
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.type", feature_match_type,
        "--FeatureMatching.num_threads", "-1",        
        "--FeatureMatching.guided_matching", guided_matching,
        "--FeatureMatching.skip_geometric_verification", "0",
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", str(sift_match_max_ratio),
        "--SiftMatching.max_distance", str(sift_match_max_distance),
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
        "--LomaMatching.min_score", str(loma_match_min_score),
        "--LomaMatching.use_bf16", str(loma_match_use_bf16),
        "--LomaMatching.b_model_path", b_model_path,
        "--LomaMatching.b_model_path_bf16", b_model_path_bf16,
        "--LomaMatching.b128_model_path", b128_model_path,
        "--LomaMatching.b128_model_path_bf16", b128_model_path_bf16,
        "--LomaMatching.r_model_path", r_model_path,
        "--LomaMatching.r_model_path_bf16", r_model_path_bf16,
        "--LomaMatching.l_model_path", l_model_path,
        "--LomaMatching.l_model_path_bf16", l_model_path_bf16,
        "--LomaMatching.g_model_path", g_model_path,
        "--LomaMatching.g_model_path_bf16", g_model_path_bf16,
        "--LomaMatching.brute_force_min_cossim", "0.85",
        "--LomaMatching.brute_force_max_ratio", "1",
        "--LomaMatching.brute_force_cross_check", "1",
        "--LomaMatching.brute_force_model_path", bruteforce_match_path,
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
        "--VocabTreeMatching.num_images_after_verification", str(min_matches_per_image), # 0
        "--VocabTreeMatching.max_num_features", str(vocab_feature_num),
        "--VocabTreeMatching.vocab_tree_path", vocab_path,
        # "--VocabTreeMatching.match_list_path", match_list_path,
        "--VocabTreeMatching.num_threads", "-1"
    ]
    
    return vocab_tree_matcher_cmd


def get_spatial_matcher_cmd(colmap_command: str,
                                log_level: int, 
                                database_path: str, 
                                feature_match_type: str, 
                                use_gpu: int,
                                sift_lightglue_match_path: str,
                                bruteforce_match_path: str,
                                aliked_lightglue_match_path: str, 
                                loma_match_min_score: float = 0.1,
                                loma_match_use_bf16: int = 0,
                                b_model_path: str = "",
                                b_model_path_bf16: str = "",
                                b128_model_path: str = "",
                                b128_model_path_bf16: str = "",
                                r_model_path: str = "",
                                r_model_path_bf16: str = "",
                                l_model_path: str = "",
                                l_model_path_bf16: str = "",
                                g_model_path: str = "",
                                g_model_path_bf16: str = "",                                                            
                                max_feature_num: int = 2048,
                                min_num_inliers: int = 15,
                                min_inlier_ratio: float = 0.25,
                                sift_match_max_distance: float = 0.7,
                                sift_match_max_ratio: float = 0.5,                             
                                two_view_geometry_max_error: float = 4.0,
                                ignore_z: int = 1,
                                max_num_neighbors: int = 300,
                                min_num_neighbors: int = 50,
                                farest_image_distance: float = 100
                                ):
    
    if feature_match_type.lower().startswith("sift"):
        guided_matching = "1"
    else:
        guided_matching = "0"
    spatial_matcher_cmd = [
        colmap_command, "spatial_matcher",
        "--log_level", str(log_level),
        "--database_path", database_path,        
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.type", feature_match_type,
        "--FeatureMatching.num_threads", "-1",        
        "--FeatureMatching.guided_matching", guided_matching,
        "--FeatureMatching.skip_geometric_verification", "0",
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(max_feature_num),
        "--SiftMatching.max_ratio", str(sift_match_max_ratio),
        "--SiftMatching.max_distance", str(sift_match_max_distance),
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
        "--LomaMatching.min_score", str(loma_match_min_score),
        "--LomaMatching.use_bf16", str(loma_match_use_bf16),
        "--LomaMatching.b_model_path", b_model_path,
        "--LomaMatching.b_model_path_bf16", b_model_path_bf16,
        "--LomaMatching.b128_model_path", b128_model_path,
        "--LomaMatching.b128_model_path_bf16", b128_model_path_bf16,
        "--LomaMatching.r_model_path", r_model_path,
        "--LomaMatching.r_model_path_bf16", r_model_path_bf16,
        "--LomaMatching.l_model_path", l_model_path,
        "--LomaMatching.l_model_path_bf16", l_model_path_bf16,
        "--LomaMatching.g_model_path", g_model_path,
        "--LomaMatching.g_model_path_bf16", g_model_path_bf16,
        "--LomaMatching.brute_force_min_cossim", "0.85",
        "--LomaMatching.brute_force_max_ratio", "1",
        "--LomaMatching.brute_force_cross_check", "1",
        "--LomaMatching.brute_force_model_path", bruteforce_match_path,        
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
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio),
        "--TwoViewGeometry.random_seed", "-1",
        "--SpatialMatching.ignore_z", str(ignore_z),
        "--SpatialMatching.max_num_neighbors", str(max_num_neighbors),
        "--SpatialMatching.min_num_neighbors", str(min_num_neighbors),
        "--SpatialMatching.max_distance", str(farest_image_distance)
    ]
    return spatial_matcher_cmd          
