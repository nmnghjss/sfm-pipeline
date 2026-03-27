import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from utils import *
import numpy as np
import cv2
from database import COLMAPDatabase, ReadColmapDatabase, filter_matches_by_inliers
from visualization import visualize_image_pairs
import ctypes

# --------------------------
# Argument parser
# --------------------------
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", default="E:\\Test1234\\data18", type=str)
parser.add_argument("--output_path", "-o", default="", type=str)
parser.add_argument("--camera", default="SIMPLE_RADIAL", type=str)
parser.add_argument("--default_focal_length_factor", default=1.0, type=float, help="Default focal length as a factor of image size (if not specified in EXIF)")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--single_camera", "-sc",default="1", type=str)
parser.add_argument("--single_fold", "-sf", default="0", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--feature_type", type=str, default="ALIKED_N16ROT", choices=["SIFT", "ALIKED_N16ROT", "ALIKED_N32"], help="Feature type for COLMAP feature extraction (e.g., SIFT, ALIKED_N16ROT, ALIKED_N32)")
parser.add_argument("--max_image_size", type=int, default=-1, help="maximum image size used to extract feature")
parser.add_argument("--match_strategy", "-ms", type=str, default="vocab_tree", choices=["exhaustive", "sequential", "vocab_tree"], help="Matching strategy to use")
parser.add_argument("--match_alg", "-ma", type=str, default="LIGHTGLUE", choices=["BRUTEFORCE", "LIGHTGLUE"], help="Matching type for COLMAP (e.g., ALIKED_LIGHTGLUE, ALIKED_N32)")
parser.add_argument("--vocab_feature_num", type=int, default=0, help="vocab tree retrial feature num")
parser.add_argument("--mapper", default="global", type=str, choices=["acc", "global", "hierarchical", "hierarchical_acc", "pose_prior"], help="Algorithm for matching and mapping: colmap / acc / global / hierarchical / hierarchical_acc / pose_prior")
parser.add_argument("--max_feature_num", "-mfn", default=2048, type=int, help="Maximum number of features to extract per image (for SuperPoint)")
parser.add_argument("--max_matches_per_image", "-mpi", type=int, default=30,
                    help="Max number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--similarity_threshold", "-st", type=float, default=0.75,
                    help="Similarity threshold for threshold-based matching strategy (0~1)")
parser.add_argument("--min_num_inliers", type=int, default=30, help="Minimum number of inliers for a valid match")
parser.add_argument("--min_inlier_ratio", type=float, default=0.1, help="Minimum inlier ratio for a valid match")
parser.add_argument("--sequential_overlap", "-so", type=int, default=15, help="Number of neighboring images to match on each side for sequential matching")
parser.add_argument("--filt_match", action="store_true", help="Whether to filter matches by inliers before mapping")
parser.add_argument("--filter_inlier_ratio_threshold", type=float, default=0.6, help="Inlier ratio threshold for filtering matches before mapping")
parser.add_argument("--filter_inlier_num_threshold", type=int, default=30, help="Inlier number threshold for filtering matches before mapping")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
parser.add_argument("--visualize_matches", "-vis", action="store_true", help="Whether to visualize matches")
parser.add_argument("--clean", action="store_true", help="Whether to clean the output directory")
args = parser.parse_args()

# ========================== Helper functions =========================
def resource_path() -> str:
    """Get absolute path for packaged or development scripts."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

# ============================== output path ============================
if args.output_path == "":
    output_path = args.source_path
else:
    output_path = args.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.source_path, output_path)
os.makedirs(output_path, exist_ok=True)


# ============================ Logging setup ===============================
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


# ============================ Timeing variables ===============================
start_time = time.time()
feature_extraction_time = 0
feature_matching_time = 0
splg_time = 0
mapper_time = 0

# =====================key parameter =====================================================
min_num_inliers = args.min_num_inliers
min_inlier_ratio = args.min_inlier_ratio

feature_match_type = args.match_alg
if feature_match_type != "UNDEFINED":
    if args.feature_type == "SIFT":
        feature_match_type = "SIFT" + "_" + feature_match_type
    else:
        feature_match_type = "ALIKED" + "_" + feature_match_type



# ===================== Paths and executables ============================================
os_type = check_operating_system()
current_path = resource_path()
print(f"Detected operating system: {os_type}")
if os_type == 'Windows':
    # colmap_path = os.path.join(current_path, "colmap-x64-windows-cuda-3.13.0/bin/colmap.exe")
    colmap_path = "D:\\Codes\\Study\\colmap\\build\\src\\colmap\\exe\\Release\\colmap.exe"
    # colmap_path = "D:\\Programs\\colmap-x64-windows-cuda-4.0.2\\bin\\colmap.exe"
else:
    colmap_path = "colmap"

colmap_command = args.colmap_executable if args.colmap_executable else colmap_path

bruteforce_match_path = os.path.join(current_path, "checkpoints/colmap_dep/bruteforce-matcher.onnx")
sift_lightglue_match_path = os.path.join(current_path, "checkpoints/colmap_dep/sift-lightglue.onnx")
aliked_lightglue_match_path = os.path.join(current_path, "checkpoints/colmap_dep/aliked-lightglue.onnx")
aliked_n16rot_path = os.path.join(current_path, "checkpoints/colmap_dep/aliked-n16rot.onnx")
aliked_n32_path = os.path.join(current_path, "checkpoints/colmap_dep/aliked-n32.onnx")
vocab_sift_path = os.path.join(current_path, "checkpoints/colmap_dep/vocab_tree_faiss_flickr100K_words256K.bin")
vocab_aliked_n16rot_path = os.path.join(current_path, "checkpoints/colmap_dep/vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin")
vocab_aliked_n32_path = os.path.join(current_path, "checkpoints/colmap_dep/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin")


if args.feature_type == "SIFT":
    vocab_path = vocab_sift_path
elif args.feature_type == "ALIKED_N16ROT":
    vocab_path = vocab_aliked_n16rot_path
elif args.feature_type == "ALIKED_N32":
    vocab_path = vocab_aliked_n32_path
else:
    vocab_path = ""



# ========================== clean output directory ==========================
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


# ======================== GPU setup ==========================================
use_gpu = 0 if args.no_gpu else 1

# ==============================================================================
# Helper function to convert Chinese paths to short paths on Windows
def get_short_path_name(long_path):
    """
    Convert a long path to its short (8.3) path format on Windows.
    This helps handle paths with non-ASCII characters (like Chinese).
    
    Args:
        long_path: The long path to convert
        
    Returns:
        Short path if on Windows, original path otherwise
    """
    if sys.platform != 'win32':
        return long_path
    
    try:
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.kernel32.GetShortPathNameW(long_path, buffer, len(buffer)):
            short_path = buffer.value
            logger.debug(f"Converted path: {long_path} -> {short_path}")
            return short_path
    except Exception as e:
        logger.warning(f"Failed to convert path to short format: {e}")
    
    return long_path  # Fallback to the original path if conversion fails

# ==============================================================================

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


# ========================= Feature extraction ==================================
logger.info("=== Starting Structure-from-Motion Pipeline ===")
logger.info(f"Source path: {args.source_path}")
logger.info(f"Output path: {output_path}")
distorted_sparse_path = os.path.join(output_path, "distorted/sparse")
os.makedirs(distorted_sparse_path, exist_ok=True)

database_path = os.path.join(output_path, "distorted/database.db")
match_list_path = os.path.join(distorted_sparse_path, "image_pairs_to_match.txt")
images_path = os.path.join(args.source_path, "input")

input_img_num, image_files = count_images_in_dir_recursive(images_path)
logger.info(f"input images num: {input_img_num}")
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = sorted([
    f for f in os.listdir(images_path)
    if os.path.splitext(f)[1].lower() in image_extensions
])


logger.info(f"Found {len(image_files)} images")

feat_extraction_cmd = [
    colmap_command, "feature_extractor",
    "--log_level", str(log_level),
    "--database_path", database_path,
    "--image_path", images_path,
    "--FeatureExtraction.type", args.feature_type, # UNDEFINED, SIFT, ALIKED_N16ROT, ALIKED_N32
    "--ImageReader.single_camera_per_image", str(args.single_image),
    "--ImageReader.single_camera_per_fold", str(args.single_fold),
    "--ImageReader.single_camera", str(args.single_camera),
    "--ImageReader.camera_model", args.camera,
    # "--ImageReader.mask_path", args.mask_path,
    # "--ImageReader.existing_camera_id", str(args.existing_camera_id),
    # "--ImageReader.camera_params", args.camera_params,
    "--ImageReader.default_focal_length_factor", str(args.default_focal_length_factor),
    # "--ImageReader.camera_mask_path", args.camera_mask_path,
    # "--FeatureExtraction.num_threads", str(args.num_threads),
    "--FeatureExtraction.use_gpu", str(use_gpu),
    "--FeatureExtraction.gpu_index","-1",
    "--FeatureExtraction.max_image_size", str(args.max_image_size),
    "--SiftExtraction.max_num_features", str(args.max_feature_num),
    # "--SiftExtraction.first_octave", str(args.first_octave),
    # "--SiftExtraction.num_octaves", str(args.num_octaves),
    # "--SiftExtraction.octave_resolution", str(args.octave_resolution),
    # "--SiftExtraction.peak_threshold", str(args.peak_threshold),
    # "--SiftExtraction.edge_threshold", str(args.edge_threshold),
    # "--SiftExtraction.estimate_affine_shape", str(args.estimate_affine_shape),
    # "--SiftExtraction.max_num_orientations", str(args.max_num_orientations),
    # "--SiftExtraction.upright", str(args.upright),
    "--SiftExtraction.domain_size_pooling", "0",
    "--SiftExtraction.dsp_min_scale", "0.167",
    "--SiftExtraction.dsp_max_scale", "3",
    "--SiftExtraction.dsp_num_scales", "10",
    "--AlikedExtraction.max_num_features", str(args.max_feature_num),
    "--AlikedExtraction.min_score", "0.2",
    "--AlikedExtraction.n16rot_model_path", aliked_n16rot_path,
    "--AlikedExtraction.n32_model_path", aliked_n32_path

]
logger.info("Starting feature extraction with COLMAP SIFT...")
t0 = time.time()
run_subprocess(feat_extraction_cmd, logger)
feature_extraction_time = time.time() - t0
logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

# --- Feature matching ---
logger.info("Starting feature matching...")
t1 = time.time()
if args.match_strategy == "exhaustive":
    feat_matching_cmd = [
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
        "--FeatureMatching.max_num_matches", str(args.max_feature_num),
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
        "--TwoViewGeometry.max_error", "4",
        "--TwoViewGeometry.confidence", "0.999",
        "--TwoViewGeometry.max_num_trials", "10000",
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--TwoViewGeometry.random_seed", "-1",
    ]
elif args.match_strategy == "sequential":
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
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--match_list_path", seq_match_list_path,
        "--match_alg", "pairs", # {'pairs', 'raw', 'inliers'}               
        "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT, ALIKED_LIGHTGLUE, ALIKED_N32
        "--FeatureMatching.num_threads", "-1",
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.guided_matching", "0",
        "--FeatureMatching.skip_geometric_verification", "0"
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", str(args.max_feature_num),
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
        "--TwoViewGeometry.max_error", "4",
        "--TwoViewGeometry.confidence", "0.999",
        "--TwoViewGeometry.max_num_trials", "10000",
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--TwoViewGeometry.random_seed", "-1"
    ]
elif args.match_strategy == "vocab_tree":
    feat_matching_cmd = [
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
        "--FeatureMatching.max_num_matches", str(args.max_feature_num),
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
        "--TwoViewGeometry.max_error", "4",
        "--TwoViewGeometry.confidence", "0.9999",
        "--TwoViewGeometry.max_num_trials", "10000",
        "--TwoViewGeometry.min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--TwoViewGeometry.random_seed", "-1",
        "--VocabTreeMatching.num_images", str(args.max_matches_per_image), # 100
        "--VocabTreeMatching.num_nearest_neighbors", "5", # 5
        "--VocabTreeMatching.num_checks", "64",
        "--VocabTreeMatching.num_images_after_verification", "0", # 0
        "--VocabTreeMatching.max_num_features", str(args.vocab_feature_num),
        "--VocabTreeMatching.vocab_tree_path", vocab_path,
        # "--VocabTreeMatching.match_list_path", match_list_path,
        "--VocabTreeMatching.num_threads", "-1"
    ]

run_subprocess(feat_matching_cmd, logger)
feature_matching_time = time.time() - t1
logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

# --------------- calibrate view graph -------------------------------------
view_graph_calibrate_cmd = [
    colmap_command, "view_graph_calibrator",
    "--log_level", str(log_level),
    "--database_path", database_path,
    "--cross_validate_prior_focal_lengths", "1",
    "--min_calibrated_pair_ratio", "0.5",
    "--reestimate_relative_pose", "1",
    "--min_focal_length_ratio", "0.1",
    "--max_focal_length_ratio", "10",
    "--max_calibration_error", "2",
    "--relpose_max_error", "1",
    "--relpose_min_num_inliers", "30",
    "--relpose_min_inlier_ratio", "0.25",
]

view_graph_calibrate_start_time = time.time()
run_subprocess(view_graph_calibrate_cmd, logger)
view_graph_calibrate_time = time.time() - view_graph_calibrate_start_time
logger.info(f"View graph calibrate done in {view_graph_calibrate_time:.2f} s")

## --------- visualize matches (optional) ---------
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


## ----------------- filt matches by inliers (optional) -----------------
if args.filt_match:
    logger.info("Filtering matches by inliers before mapping...")
    filted_paris_num = filter_matches_by_inliers(
        database_path=database_path,
        min_num_inliers=args.filter_inlier_num_threshold,
        min_inlier_ratio=args.filter_inlier_ratio_threshold,
        logger=logger
    )
    logger.info(f"Filtering done. removed pairs: {filted_paris_num}")

# --------------------------
# Mapper / Bundle Adjustment
# --------------------------
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper...")
t2 = time.time()
if args.mapper == "acc":
    mapper_cmd = [
        colmap_command, "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.num_threads", "-1",
        "--Mapper.min_num_matches", "15", # 15
        "--Mapper.init_num_trials", "1000", # 200
        "--Mapper.init_min_num_inliers", "200", # 100
        "--Mapper.init_max_error", "4", # 4
        "--Mapper.init_min_tri_angle", "16", # 16
        "--Mapper.ba_local_min_tri_angle", "6", # 6
        "--Mapper.ba_local_num_images", "6", # 6
        "--Mapper.ba_local_max_num_iterations", "25", # 25
        "--Mapper.ba_local_max_refinements", "2", # 2
        "--Mapper.ba_local_max_refinement_change", "0.001", # 0.001
        "--Mapper.ba_global_frames_ratio", "1.5", # 1.1
        "--Mapper.ba_global_points_ratio", "1.5", # 1.1
        "--Mapper.ba_global_frames_freq", "500", # 500
        "--Mapper.ba_global_points_freq", "250000", # 250000
        "--Mapper.ba_global_max_num_iterations", "50", # 50 --> 20 --> 25
        "--Mapper.ba_global_max_refinements", "5", # 5
        "--Mapper.ba_global_max_refinement_change", "0.0005", # 0.0005
        "--Mapper.ba_refine_focal_length", "1", # 1
        "--Mapper.ba_refine_principal_point", "0", # 0
        "--Mapper.ba_refine_extra_params", "1", # 1
        "--Mapper.ba_use_gpu", "0",  # 0
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
        "--image_overlap", "50", # 50
        "--leaf_max_num_images", "500", # 500
        "--Mapper.min_num_matches", "15",
        "--Mapper.ignore_watermarks", "0",
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "50",
        "--Mapper.max_model_overlap", "20",
        "--Mapper.min_model_size", "10",
        "--Mapper.init_num_trials", "1000", # 200
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", "-1",
        "--Mapper.random_seed", "-1",
        "--Mapper.min_focal_length_ratio", "0.1",
        "--Mapper.max_focal_length_ratio", "10",
        "--Mapper.max_extra_param", "1",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_refine_sensor_from_rig", "0",
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
        "--Mapper.ba_use_gpu", str(use_gpu),
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
        "--Mapper.ba_use_gpu", "0", # 0
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

elif args.mapper == "global":
    mapper_cmd = [
        colmap_command, "global_mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--GlobalMapper.image_list_path", "",
        "--GlobalMapper.min_num_matches", str(min_num_inliers),
        "--GlobalMapper.ignore_watermarks", "0",
        "--GlobalMapper.num_threads", "-1",
        "--GlobalMapper.random_seed", "-1",
        "--GlobalMapper.decompose_relative_pose", "1",
        "--GlobalMapper.ba_num_iterations", "3", # 3
        "--GlobalMapper.skip_rotation_averaging", "0",
        "--GlobalMapper.skip_track_establishment", "0",
        "--GlobalMapper.skip_global_positioning", "0",
        "--GlobalMapper.skip_bundle_adjustment", "0",
        "--GlobalMapper.skip_retriangulation", "0",
        "--GlobalMapper.track_intra_image_consistency_threshold", "10",
        "--GlobalMapper.track_required_tracks_per_view", "2147483647",
        "--GlobalMapper.track_min_num_views_per_track", "3",
        "--GlobalMapper.gp_use_gpu", "0", # 1
        "--GlobalMapper.gp_gpu_index", "-1",
        "--GlobalMapper.gp_optimize_positions", "1",
        "--GlobalMapper.gp_optimize_points", "1",
        "--GlobalMapper.gp_optimize_scales", "1",
        "--GlobalMapper.gp_loss_function_scale", "0.1",
        "--GlobalMapper.gp_max_num_iterations", "100", # 100
        "--GlobalMapper.ba_refine_focal_length", "1",
        "--GlobalMapper.ba_refine_principal_point", "0",
        "--GlobalMapper.ba_refine_extra_params", "1",
        "--GlobalMapper.ba_refine_sensor_from_rig", "0",
        "--GlobalMapper.ba_refine_rig_from_world", "1",
        "--GlobalMapper.ba_refine_points3D", "1",
        "--GlobalMapper.ba_min_track_length", "3",
        "--GlobalMapper.ba_ceres_use_gpu", "0", # 1
        "--GlobalMapper.ba_ceres_gpu_index", "-1",
        "--GlobalMapper.ba_ceres_loss_function_scale", "1",
        "--GlobalMapper.ba_ceres_max_num_iterations", "200",
        "--GlobalMapper.ba_skip_fixed_rotation_stage", "0",
        "--GlobalMapper.ba_skip_joint_optimization_stage", "0",
        "--GlobalMapper.tri_complete_max_reproj_error", "15",
        "--GlobalMapper.tri_merge_max_reproj_error", "15",
        "--GlobalMapper.tri_min_angle", "1", #1
        "--GlobalMapper.ra_max_rotation_error_deg", "10",
        "--GlobalMapper.max_angular_reproj_error_deg", "1",
        "--GlobalMapper.max_normalized_reproj_error", "0.01",
        "--GlobalMapper.min_tri_angle_deg", "1", #1
    ]

elif args.mapper == "pose_prior":
    mapper_cmd = [
        colmap_command, "pose_prior_mapper",
        "--log_level", "0",
        "--log_severity", "0",
        "--log_color", "1",
        "--database_path", database_path,
        "--image_path", images_path,
        "--input_path", "",
        "--output_path", distorted_sparse_path,
        "--Mapper.min_num_matches", str(min_num_inliers),
        "--Mapper.ignore_watermarks", "0",
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", "-1",
        "--Mapper.random_seed", "-1",
        "--Mapper.ba_use_gpu", "0", # 0
        "--Mapper.ba_gpu_index", "-1",        
        "--Mapper.min_focal_length_ratio", "0.1",
        "--Mapper.max_focal_length_ratio", "10",
        "--Mapper.max_extra_param", "1",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_refine_sensor_from_rig", "1",
        "--Mapper.ba_local_function_tolerance", "0",
        "--Mapper.ba_local_max_num_iterations", "25",
        "--Mapper.ba_global_frames_ratio", "1.5", # 1.1
        "--Mapper.ba_global_points_ratio", "1.5", # 1.1
        "--Mapper.ba_global_frames_freq", "500",
        "--Mapper.ba_global_points_freq", "250000",
        "--Mapper.ba_global_function_tolerance", "0",
        "--Mapper.ba_global_max_num_iterations", "25", # 50
        "--Mapper.ba_global_max_refinements", "2", # 5
        "--Mapper.ba_global_max_refinement_change", "0.001", # 0.0005
        "--Mapper.ba_local_max_refinements", "2", # 2
        "--Mapper.ba_local_max_refinement_change", "0.001", # 0.001
        "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        "--Mapper.snapshot_path", "",
        "--Mapper.snapshot_frames_freq", "0",
        "--Mapper.fix_existing_frames", "0",
        "--Mapper.init_min_num_inliers", "1000", # 100
        "--Mapper.init_max_error", "4",
        "--Mapper.init_max_forward_motion", "0.95",
        "--Mapper.init_min_tri_angle", "16",
        "--Mapper.init_max_reg_trials", "2",
        "--Mapper.abs_pose_max_error", "12",
        "--Mapper.abs_pose_min_num_inliers", str(min_num_inliers), # 30
        "--Mapper.abs_pose_min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--Mapper.filter_max_reproj_error", "4",
        "--Mapper.filter_min_tri_angle", "1.5",
        "--Mapper.max_reg_trials", "3",
        "--Mapper.ba_local_num_images", "6", # 6
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
        "--Mapper.tri_ignore_two_view_tracks", "1",
        "--overwrite_priors_covariance", "0",
        "--prior_position_std_x", "1",
        "--prior_position_std_y", "1",
        "--prior_position_std_z", "1",
        "--use_robust_loss_on_prior_position", "0",
        "--prior_position_loss_scale", "7.82",
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

img_num, _ = count_images_in_dir_recursive(os.path.join(output_path, "images"))

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
logger.info(f"  View Grpha Calibrate time: {view_graph_calibrate_time} s")
logger.info(f"  Mapper: {int(mapper_time // 60)} min {mapper_time % 60:.2f} s")
logger.info(f"  Sparse reconstruction: {int(sparse_reconstruction_time // 60)} min {sparse_reconstruction_time % 60:.2f} s")
logger.info(f"  Image undistortion: {int(undistort_time // 60)} min {undistort_time % 60:.2f} s")
logger.info(f"  Total time: {int(total_time // 60)} min {total_time % 60:.2f} s")
logger.info(f"  Number of images registered: {img_num} / {input_img_num}")