
import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from match_cmd import get_matches_importer_cmd
from utils import *
from pathlib import Path
import torch
import numpy as np
import cv2
import sqlite3
from lightglue import ALIKED, LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, load_image_use_torchvision, safe_load_image, load_image_use_PIL
from database import COLMAPDatabase, ReadColmapDatabase
from visualization import visualize_image_pairs


from match_utils import compute_matched_image_pairs_by_pose_prior
from database import initialize_colmap_database
from feature_extractor_cmd import extract_neural_features
from match_cmd import match_features_with_lightglue


# ============================= Argument parser ===========================================
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", default="E:\\Test1234\\data18", type=str)
parser.add_argument("--output_path", "-o", default="", type=str)
parser.add_argument("--camera", default="SIMPLE_RADIAL", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--single_camera", "-sc",default="1", type=str)
parser.add_argument("--single_fold", "-sf", default="0", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--slam_pose_path", type=str, help="Path to SLAM pose file (if available)")
parser.add_argument("--mapper", default="acc", type=str, help="Algorithm for matching and mapping: colmap / acc / glomap")
parser.add_argument("--feature_type", "-ft", default="superpoint", type=str, choices=["superpoint", "aliked", "disk"], help="Feature type to use: superpoint or aliked")
parser.add_argument("--max_feature_num", "-mfn", default=2048, type=int, help="Maximum number of features to extract per image (for SuperPoint)")
parser.add_argument("--SuperpointLightglue", "-splg", action="store_true", help="Use SuperPoint features instead of SIFT")
parser.add_argument("--match_strategy", "-ms", type=str, default="threshold", 
                    choices=["exhaustive", "nearest_k", "quick", "threshold"],
                    help="Matching strategy: exhaustive (all pairs), nearest_k (top-k similar), quick (fast heuristic), threshold (similarity threshold based)")
parser.add_argument("--max_matches_per_image", "-mpi", type=int, default=30,
                    help="Max number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--min_matches_per_image", "-mni", type=int, default=10,
                    help="Minimum number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--similarity_threshold", "-st", type=float, default=0.75,
                    help="Similarity threshold for threshold-based matching strategy (0~1)")
parser.add_argument("--min_num_inliers", type=int, default=30, help="Minimum number of inliers for a valid match")
parser.add_argument("--min_inlier_ratio", type=float, default=0.1, help="Minimum inlier ratio for a valid match")
parser.add_argument("--sift_match_strategy", "-sms", type=str, default="acc", choices=["default","acc", "sequential", "vocab_tree"], help="Matching strategy for SIFT features")
parser.add_argument("--sequential_overlap", "-so", type=int, default=15, help="Number of neighboring images to match on each side for sequential matching")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
parser.add_argument("--visualize_matches", "-vis", action="store_true", help="Whether to visualize matches")
parser.add_argument("--clean", action="store_true", help="clean the output dir before sfm pipelene")
args = parser.parse_args()

# args.SuperpointLightglue = True  # 强制使用 SuperPoint + LightGlue

# =====================key parameter =====================================================
min_num_inliers = args.min_num_inliers
min_inlier_ratio = args.min_inlier_ratio

# ========================= Helper functions =============================================
def resource_path() -> str:
    """Get absolute path for packaged or development scripts."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

# ========================== Paths and executables =======================================
current_path = resource_path()
os_type = check_operating_system()
print(f"Detected operating system: {os_type}")
if os_type == 'Windows':
    # colmap_path = os.path.join(current_path, "colmap-x64-windows-cuda-3.13.0/bin/colmap.exe")
    # colmap_path = "D:\\Codes\\Work\\colmap\\build\\src\\colmap\\exe\\Release\\colmap.exe"
    colmap_path = "D:\\Programs\\colmap-x64-windows-cuda-4.0.2\\bin\\colmap.exe"
    glomap_path = os.path.join(current_path, "glomap-x64-windows-cuda-1.2.0\\bin\\glomap.exe")
    vocab_path = os.path.join(current_path, "colmap-x64-windows-cuda-3.13.0/vocab/vocab_tree_faiss_flickr100K_words256K.bin")
else:
    colmap_path = "colmap"
    glomap_path = "glomap"
    vocab_path = os.path.join(current_path, "vocab/vocab_tree_faiss_flickr100K_words256K.bin")
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

distorted_sparse_path = os.path.join(output_path, "distorted/sparse")
os.makedirs(distorted_sparse_path, exist_ok=True)

database_path = os.path.join(output_path, "distorted/database.db")
images_path = os.path.join(args.source_path, "input")
matched_images_pairs_path = os.path.join(distorted_sparse_path, "image_pairs.txt")
input_img_num, _ = count_images_in_dir_recursive(images_path)

if args.clean:
    print(f"Cleaning output directory: {output_path}")
    try:
        shutil.rmtree(os.path.join(output_path, "distorted"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_path, "sparse"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_path, "dense"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_path, "stereo"), ignore_errors=True)
        print("Output directory cleaned successfully.")
    except Exception as e:
        print(f"Failed to clean output directory: {e}")    

# =========================== Logging setup ==========================
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

logger.info("use_gpu: {}".format(use_gpu))

# ============================ Timing variables ====================================
start_time = time.time()
feature_extraction_time = 0
feature_matching_time = 0
splg_time = 0
mapper_time = 0

# ======================== SuperPoint + LightGlue Functions ==============================

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


# ============================ find matched pairs using slam pose (optional) ==========
if args.slam_pose_path is not None and len(args.slam_pose_path) > 0:
    logger.info(f"Finding image pairs based on SLAM poses from: {args.slam_pose_path}")
    # pcd=load_point_cloud(os.path.join(args.slam_pose_path, "scan_all_pointcloud.ply"))
    # poses=load_poses(os.path.join(args.slam_pose_path, "scan_all_pose.ply"))
    # intrinsics=load_intrinsics(os.path.join(args.slam_pose_path, "scan_camera_intrinsics.txt"))
    # build_frustum_overlap_voxel(
    #     pcd=pcd,
    #     poses=poses,
    #     intrinsics=intrinsics,
    #     output_txt=matched_images_pairs_path,
    #     voxel_size=0.1,
    #     max_pairs_per_image=args.max_matches_per_image,
    #     overlap_thresh=0.5,
    #     max_angle=np.radians(45)  # 45度视角约束
    # )

    compute_matched_image_pairs_by_pose_prior(args.slam_pose_path, matched_images_pairs_path, overlap_thresh=0.5)
    logger.info(f"SLAM pose-based image pairs saved to: {matched_images_pairs_path}")


# ====================== Feature extraction & matching ===================================
logger.info("=== Starting Structure-from-Motion Pipeline ===")
logger.info(f"Source path: {args.source_path}")
logger.info(f"Output path: {output_path}")

# --- Feature extraction ---
if args.SuperpointLightglue:
    logger.info("Using SuperPoint + LightGlue for feature extraction and matching...")
    logger.info(f"  Match strategy: {args.match_strategy}")
    if args.match_strategy in ["nearest_k", "quick"]:
        logger.info(f"  Max matches per image: {args.max_matches_per_image}")
    
    # Initialize COLMAP database with images (without SIFT extraction)
    logger.info("Initializing COLMAP database with image metadata (without SIFT extraction)...")
    splg_start = time.time()
    
    db_init_success = initialize_colmap_database(
        database_path=database_path,
        images_path=images_path,
        camera_model=args.camera,
        logger=logger
    )
    
    if not db_init_success:
        logger.error("Failed to initialize database!")
        sys.exit(1)
    
    # Now run SuperPoint feature extraction    
    image_features, image_id_map, feat_ext_time = extract_neural_features(
        args.feature_type,
        current_path,
        database_path, 
        images_path, 
        max_num_keypoints=args.max_feature_num,
        logger=logger
    )
    feature_extraction_time = time.time() - splg_start
    logger.info(f"SuperPoint feature extraction time: {feature_extraction_time:.2f} s (including database writes)")
    
    # Run LightGlue feature matching
    feat_match_time = match_features_with_lightglue(
        current_path,
        args.feature_type,
        database_path,
        image_features,
        image_id_map,
        match_list_path=matched_images_pairs_path,
        match_strategy=args.match_strategy,
        max_matches_per_image=args.max_matches_per_image,
        min_matches_per_image=args.min_matches_per_image,
        similarity_threshold=args.similarity_threshold,
        logger=logger
    )
    feature_matching_time = feat_match_time
    splg_time = time.time() - splg_start
    
    # --- Import matches using COLMAP matches_importer ---
    # This is the recommended way to import pre-computed matches into COLMAP database
    # Following the same approach as super_colmap.py
    logger.info("Importing matches into COLMAP database using matches_importer...")
    logger.info(f"Match list path: {matched_images_pairs_path}")
    logger.info(f"Database path: {database_path}")
    matches_importer_cmd = get_matches_importer_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        feature_match_type="SIFT_BRUTEFORCE",  # Use brute-force matcher for imported matches
        database_path=database_path,
        matched_images_pairs_path=matched_images_pairs_path,
        min_num_inliers=args.min_num_inliers,
        min_inlier_ratio=args.min_inlier_ratio
    )

    run_subprocess(matches_importer_cmd, logger)
    logger.info("Matches imported successfully")
else:
    # Use COLMAP's built-in feature extractor
    
    # Get list of images first (needed for sequential matching and feature extraction)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(images_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])
    
    if not image_files:
        logger.error(f"No images found in {images_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_files)} images")
    
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_path,
        "--ImageReader.single_camera_per_image", str(args.single_image),
        "--ImageReader.single_camera_per_fold", str(args.single_fold),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.camera,
        # "--SiftExtraction.max_num_features", str(args.max_feature_num),
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
    if args.sift_match_strategy == "default":
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--log_level", str(log_level),
        ]
    elif args.sift_match_strategy == "acc":
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--ExhaustiveMatching.block_size", "200", # 50
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.guided_matching", "0", # 0
            # "--FeatureMatching.max_num_matches", str(args.max_feature_num), # 32768
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
    elif args.sift_match_strategy == "sequential":
        # Generate sequential match list with circular overlap
        logger.info("Generating sequential match pairs...")
        
        # Get image names from image_files
        image_names = [os.path.basename(f) for f in image_files]
        
        # Define overlap parameter from command-line argument
        seq_overlap = args.sequential_overlap
        
        # Generate match pairs and save to file
        seq_match_list_path = os.path.join(output_path, "sequential_match_list.txt")
        match_pairs = generate_sequential_match_list(
            image_names=image_names,
            overlap=seq_overlap,
            output_file=seq_match_list_path,
            logger=logger
        )
        
        logger.info(f"Generated {len(match_pairs)} sequential match pairs (overlap={seq_overlap})")
        
        # Use matches_importer to import the sequential match list
        feat_matching_cmd = [
            colmap_command, "matches_importer",
            "--database_path", database_path,
            "--match_list_path", seq_match_list_path,
            "--match_type", "pairs",
            "--log_level", str(log_level),
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.gpu_index", "-1",
            "--FeatureMatching.guided_matching", "0",
            "--FeatureMatching.rig_verification", "0",
            # "--FeatureMatching.max_num_matches", str(args.max_feature_num), # 32768
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check", "1",
            "--SiftMatching.cpu_brute_force_matcher", "0",
            "--TwoViewGeometry.min_num_inliers", str(min_num_inliers), # 15
            "--TwoViewGeometry.multiple_models", "0",
            "--TwoViewGeometry.compute_relative_pose", "0",
            "--TwoViewGeometry.detect_watermark", "0",
            "--TwoViewGeometry.multiple_ignore_watermark", "1",
            "--TwoViewGeometry.watermark_detection_max_error", "4",
            "--TwoViewGeometry.filter_stationary_matches", "0",
            "--TwoViewGeometry.stationary_matches_max_error", "4",
            "--TwoViewGeometry.max_error", "4",
            "--TwoViewGeometry.confidence", "0.999",
            "--TwoViewGeometry.max_num_trials", "10000",
            "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
            "--TwoViewGeometry.random_seed", "-1",
        ]
    elif args.sift_match_strategy == "vocab_tree":
        feat_matching_cmd = [
            colmap_command, "vocab_tree_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--log_level", str(log_level),
            "--FeatureMatching.type", "SIFT",
            "--FeatureMatching.num_threads", str(args.num_threads), # -1
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.gpu_index", str(args.gpu_index), # -1
            "--FeatureMatching.guided_matching", "1",
            "--FeatureMatching.rig_verification", "0",
            "--FeatureMatching.max_num_matches", str(args.max_feature_num), # 32768
            "--SiftMatching.max_ratio", "0.8",
            "--SiftMatching.max_distance", "0.7",
            "--SiftMatching.cross_check", "1",
            "--SiftMatching.cpu_brute_force_matcher", "0",
            "--TwoViewGeometry.min_num_inliers", str(min_num_inliers), # 15
            "--TwoViewGeometry.multiple_models", "0",
            "--TwoViewGeometry.compute_relative_pose", "0",
            "--TwoViewGeometry.detect_watermark", "0",
            "--TwoViewGeometry.multiple_ignore_watermark", "1",
            "--TwoViewGeometry.watermark_detection_max_error", "4",
            "--TwoViewGeometry.filter_stationary_matches", "0",
            "--TwoViewGeometry.stationary_matches_max_error", "4",
            "--TwoViewGeometry.max_error", "4",
            "--TwoViewGeometry.confidence", "0.999",
            "--TwoViewGeometry.max_num_trials", "10000",
            "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
            "--TwoViewGeometry.random_seed", "-1",
            "--VocabTreeMatching.num_images", "100",
            "--VocabTreeMatching.num_nearest_neighbors", "5",
            "--VocabTreeMatching.num_checks", "64",
            "--VocabTreeMatching.num_images_after_verification", "0",
            "--VocabTreeMatching.max_num_features", "-1"
        ]

    run_subprocess(feat_matching_cmd, logger)
    feature_matching_time = time.time() - t1
    logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

#=============================== visualize matches (optional) ================================
if args.visualize_matches:
    logger.info("Visualizing matches for a few image pairs...")
    view_graph, cameras, images, feature_name = ReadColmapDatabase(database_path)
    if view_graph is None or cameras is None or images is None:
        logger.warning("Could not read database for visualization, skipping visualization step")
    else:
        vis_dir = os.path.join(output_path, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        print("vis_dir: ", vis_dir)
        visualize_image_pairs(view_graph, images, images_path, vis_dir, num_pairs=100)
        print("Visualization completed.")

# =========================== Mapper / Bundle Adjustment ===========================
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper...")
t2 = time.time()

if args.slam_pose_path is not None and len(args.slam_pose_path) > 0:
    logger.info("Using SLAM-based mapper (experimental, may not work well)...")

    triangulate_cmd = [
        colmap_command, "point_triangulator",
        "--database_path", database_path,
        "--image_path", images_path,
        "--input_path", args.slam_pose_path,
        "--output_path", distorted_sparse_path,
        "--log_level", str(log_level),
        "--clear_points", "1",              
        "--refine_intrinsics", "0", # 0
        "--Mapper.min_num_matches", str(min_num_inliers), # 15
        # "--Mapper.ignore_watermarks", "1", # 0
        # "--Mapper.multiple_models", "1",
        # "--Mapper.max_num_models", "50",
        # "--Mapper.max_model_overlap", "20",
        # "--Mapper.min_model_size", "10",
        # "--Mapper.init_image_id1", "-1",
        # "--Mapper.init_image_id2", "-1",
        # "--Mapper.init_num_trials", "1000", # 200
        # "--Mapper.structure_less_registration_fallback", "1",
        # "--Mapper.structure_less_registration_only", "0",
        # "--Mapper.extract_colors", "1",
        # "--Mapper.num_threads", "-1",
        # "--Mapper.random_seed", "-1",
        # "--Mapper.min_focal_length_ratio", "0.1",
        # "--Mapper.max_focal_length_ratio", "10",
        # "--Mapper.max_extra_param", "1",
        # "--Mapper.ba_refine_focal_length", "1",
        # "--Mapper.ba_refine_principal_point", "0",
        # "--Mapper.ba_refine_extra_params", "1",
        # "--Mapper.ba_refine_sensor_from_rig", "1",
        # "--Mapper.ba_local_function_tolerance", "0",
        # "--Mapper.ba_local_max_num_iterations", "25",
        # "--Mapper.ba_global_frames_ratio", "1.1",
        # "--Mapper.ba_global_points_ratio", "1.1",
        # "--Mapper.ba_global_frames_freq", "500",
        # "--Mapper.ba_global_points_freq", "250000",
        # "--Mapper.ba_global_function_tolerance", "0",
        # "--Mapper.ba_global_max_num_iterations", "50",
        # "--Mapper.ba_global_max_refinements", "5",
        # "--Mapper.ba_global_max_refinement_change", "0.0005",
        # "--Mapper.ba_local_max_refinements", "2",
        # "--Mapper.ba_local_max_refinement_change", "0.001",
        # "--Mapper.ba_use_gpu", "0",
        # "--Mapper.ba_gpu_index", "-1",
        # "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        # "--Mapper.snapshot_path", "snapshot",
        # "--Mapper.snapshot_frames_freq", "0",
        # "--Mapper.fix_existing_frames", "0",
        # "--Mapper.init_min_num_inliers", "100",
        # "--Mapper.init_max_error", "4",
        # "--Mapper.init_max_forward_motion", "0.95",
        # "--Mapper.init_min_tri_angle", "16",
        # "--Mapper.init_max_reg_trials", "2",
        # "--Mapper.abs_pose_max_error", "12",
        # "--Mapper.abs_pose_min_num_inliers", "30",
        # "--Mapper.abs_pose_min_inlier_ratio", "0.25",
        # "--Mapper.filter_max_reproj_error", "4",
        # "--Mapper.filter_min_tri_angle", "1.5",
        # "--Mapper.max_reg_trials", "3",
        # "--Mapper.ba_local_num_images", "6",
        # "--Mapper.ba_local_min_tri_angle", "6",
        # "--Mapper.ba_global_ignore_redundant_points3D", "0",
        # "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", "0.05",
        # "--Mapper.image_list_path",
        # "--Mapper.constant_rig_list_path",
        # "--Mapper.constant_camera_list_path",
        # "--Mapper.max_runtime_seconds", "-1",
        # "--Mapper.tri_max_transitivity", "1",
        # "--Mapper.tri_create_max_angle_error", "2",
        # "--Mapper.tri_continue_max_angle_error", "2",
        # "--Mapper.tri_merge_max_reproj_error", "4",
        # "--Mapper.tri_complete_max_reproj_error", "4",
        # "--Mapper.tri_complete_max_transitivity", "5",
        # "--Mapper.tri_re_max_angle_error", "5",
        # "--Mapper.tri_re_min_ratio", "0.2",
        # "--Mapper.tri_re_max_trials", "1",
        # "--Mapper.tri_min_angle", "1.5",
        # "--Mapper.tri_ignore_two_view_tracks", "1"
    ]
    triangulate_start = time.time()
    run_subprocess(triangulate_cmd, logger)
    triangulate_time = time.time() - triangulate_start
    logger.info(f"Triangulation completed in {triangulate_time:.2f} s")
    print("pause")
    a = input()


    refined_slam_sparse_path = os.path.join(output_path, "distorted/slam_refined_sparse")
    os.makedirs(refined_slam_sparse_path, exist_ok=True)
    ba_cmd = [
        colmap_command, "bundle_adjuster",
        "--input_path", distorted_sparse_path,
        "--output_path", distorted_sparse_path,
        "--log_level", str(log_level),
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "1",
        "--BundleAdjustment.refine_rig_from_world", "1",
        "--BundleAdjustment.refine_sensor_from_rig", "1",
        "--BundleAdjustment.refine_points3D", "1",
        "--BundleAdjustment.constant_rig_from_world_rotation", "0",
        "--BundleAdjustment.min_track_length", "0",
        "--BundleAdjustmentCeres.max_num_iterations", "100",
        "--BundleAdjustmentCeres.max_linear_solver_iterations", "200",
        "--BundleAdjustmentCeres.function_tolerance", "0",
        "--BundleAdjustmentCeres.gradient_tolerance", "0.0001",
        "--BundleAdjustmentCeres.parameter_tolerance", "0",
        "--BundleAdjustmentCeres.use_gpu", "0",
        "--BundleAdjustmentCeres.gpu_index", "-1",
        "--BundleAdjustmentCeres.min_num_images_gpu_solver", "50",
        "--BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading", "50000",
        "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver", "50",
        "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver", "1000",
        "--BundleAdjustmentCeres.max_num_images_direct_dense_gpu_solver", "200",
        "--BundleAdjustmentCeres.max_num_images_direct_sparse_gpu_solver", "4000"
    ]
    ba_start = time.time()
    for attempt in range(5):
        logger.info(f"Bundle adjustment attempt {attempt+1}/3...")
        run_subprocess(ba_cmd, logger)
    ba_time = time.time() - ba_start
    logger.info(f"Bundle adjustment completed in {ba_time:.2f} s")
else:
    if args.mapper == "acc":
        mapper_cmd = [
            colmap_command, "mapper",
            "--database_path", database_path,
            "--image_path", images_path,
            "--output_path", distorted_sparse_path,
            "--Mapper.num_threads", "-1",
            "--Mapper.min_num_matches", "25", # 15
            "--Mapper.init_num_trials", "1000", # 200
            "--Mapper.init_min_num_inliers", "100", # 100
            "--Mapper.init_max_error", "4", # 4
            "--Mapper.init_min_tri_angle", "16", # 16
            "--Mapper.ba_local_min_tri_angle", "6", # 6
            "--Mapper.ba_local_num_images", "6", # 6
            "--Mapper.ba_local_max_num_iterations", "12", # 25
            "--Mapper.ba_local_max_refinements", "2", # 2
            "--Mapper.ba_local_max_refinement_change", "0.001", # 0.001
            "--Mapper.ba_global_frames_ratio", "1.5", # 1.1
            "--Mapper.ba_global_points_ratio", "1.5", # 1.1
            "--Mapper.ba_global_frames_freq", "5000", # 5000
            "--Mapper.ba_global_points_freq", "250000", # 250000
            "--Mapper.ba_global_max_num_iterations", "25", # 50 --> 20 --> 25
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
    elif args.mapper == "hierarchical":
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

    elif args.mapper == "hierarchical_acc":
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

    elif args.mapper == "glomap":
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

# =========================== Image undistortion ===========================
start_undistort = time.time()

if args.slam_pose_path is not None and len(args.slam_pose_path) > 0:
    largest_sparse_folder = distorted_sparse_path
else:
    largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
img_undist_cmd = [
    colmap_command, "image_undistorter",
    "--image_path", images_path,
    "--input_path", largest_sparse_folder,
    "--output_path", output_path,
    "--output_type", "COLMAP"
]
run_subprocess(img_undist_cmd, logger)
undistort_time = time.time() - start_undistort
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

img_num, _ = count_images_in_dir_recursive(os.path.join(output_path, "images"))

logger.info("Sparse output successfully organized into sparse/0.")

# =========================== Timing summary ===========================
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