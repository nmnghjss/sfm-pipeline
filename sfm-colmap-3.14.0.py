
import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from utils import *
from pathlib import Path
import numpy as np
import cv2
from database import COLMAPDatabase, ReadColmapDatabase, filter_matches_by_inliers
from visualization import visualize_image_pairs

# --------------------------
# Argument parser
# --------------------------
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", default="E:\\Test1234\\data18", type=str)
parser.add_argument("--output_path", "-o", default="output_splg", type=str)
parser.add_argument("--camera", default="RADIAL", type=str)
parser.add_argument("--default_focal_length_factor", default=0.9, type=float, help="Default focal length as a factor of image size (if not specified in EXIF)")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--single_camera", "-sc",default="1", type=str)
parser.add_argument("--single_fold", "-sf", default="0", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--feature_type", type=str, default="ALIKED_N16ROT", choices=["SIFT", "ALIKED_N16ROT", "ALIKED_N32"], help="Feature type for COLMAP feature extraction (e.g., SIFT, ALIKED_N16ROT, ALIKED_N32)")
parser.add_argument("--match_strategy", "-ms", type=str, default="vocab_tree", choices=["exhaustive", "sequential", "vocab_tree"], help="Matching strategy to use")
parser.add_argument("--match_type", "-mt", type=str, default="LIGHTGLUE", choices=["UNDEFINED", "BRUTEFORCE", "LIGHTGLUE"], help="Matching type for COLMAP (e.g., ALIKED_LIGHTGLUE, ALIKED_N32)")
parser.add_argument("--alg", default="acc", type=str, help="Algorithm for matching and mapping: colmap / acc / glomap")
parser.add_argument("--max_feature_num", "-mfn", default=8192, type=int, help="Maximum number of features to extract per image (for SuperPoint)")
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


# =====================key parameter =====================================================
min_num_inliers = args.min_num_inliers
min_inlier_ratio = args.min_inlier_ratio

feature_match_type = args.match_type
if feature_match_type != "UNDEFINED":
    if args.feature_type == "SIFT":
        feature_match_type = "SIFT" + "_" + feature_match_type
    else:
        feature_match_type = "ALIKED" + "_" + feature_match_type

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
    colmap_path = "D:\\Codes\\Study\\colmap\\build\\src\\colmap\\exe\\Release\\colmap.exe"
    # colmap_path = "D:\\Programs\\colmap-x64-windows-cuda-3.13.0\\bin\\colmap.exe"
    glomap_path = os.path.join(current_path, "glomap-x64-windows-cuda-1.2.0\\bin\\glomap.exe")
    vocab_path = os.path.join(current_path, "colmap-x64-windows-cuda-3.13.0/vocab/vocab_tree_faiss_flickr100K_words32K.bin")
else:
    colmap_path = "colmap"
    glomap_path = "glomap"
    vocab_path = os.path.join(current_path, "vocab/vocab_tree_faiss_flickr100K_words32K.bin")
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

def initialize_colmap_database(
    database_path: str,
    images_path: str,
    camera_model: str = "OPENCV",
    logger: logging.Logger = None
) -> bool:
    """
    Initialize COLMAP database and add image metadata without extracting SIFT features.
    This is more efficient than using COLMAP's feature_extractor when you plan to use
    custom feature extractors like SuperPoint.
    
    Args:
        database_path: Path to create/initialize COLMAP database
        images_path: Path to images directory
        camera_model: Camera model (e.g., "OPENCV", "PINHOLE")
        args: Arguments object with camera parameters
        logger: Optional logger instance
    
    Returns:
        True if successful, False otherwise
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        logger.info("Initializing COLMAP database without SIFT extraction...")
        
        # Create/connect to database
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        
        # Get image list
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in os.listdir(images_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        
        if not image_files:
            logger.error(f"No images found in {images_path}")
            db.close()
            return False
        
        logger.info(f"Found {len(image_files)} images")
        
        # Add single camera to database (using first image dimensions)
        first_image_path = os.path.join(images_path, image_files[0])
        img = cv2.imread(first_image_path)
        if img is None:
            logger.error(f"Failed to load first image: {first_image_path}")
            db.close()
            return False
        
        height, width = img.shape[:2]
        logger.info(f"Image dimensions: {width}x{height}")
        
        # Map camera model string to COLMAP model ID    
        camera_model_id = {
            "SIMPLE_PINHOLE": 0,
            "PINHOLE": 1,
            "SIMPLE_RADIAL": 2,
            "RADIAL": 3,
            "OPENCV": 4,
            "OPENCV_FISHEYE": 5,
            "FULL_OPENCV": 6,
            "FOV": 7,
            "SIMPLE_RADIAL_FISHEYE":8,
            "RADIAL_FISHEYE":9,
            "THIN_PRISM_FISHEYE":10,
            "RAD_TAN_THIN_PRISM_FISHEYE":11
        }.get(camera_model.upper(), 4)  # Default to OPENCV

        if camera_model_id is None:
            logger.warning(f"Unknown camera model '{camera_model}', defaulting to OPENCV")
            camera_model_id = 4

        if camera_model_id == 0:
            # SIMPLE_PINHOLE: f, cx, cy
            camera_params = [max(width, height), width / 2, height / 2]
        elif camera_model_id == 1:
            # PINHOLE: fx, fy, cx, cy
            camera_params = [max(width, height), max(width, height), width / 2, height / 2]
        elif camera_model_id == 2:
            # SIMPLE_RADIAL: f, cx, cy, k
            camera_params = [max(width, height), width / 2, height / 2, 0.0]
        elif camera_model_id == 3:
            # RADIAL: f, cx, cy, k1, k2
            camera_params = [max(width, height), width / 2, height / 2, 0.0, 0.0]
        elif camera_model_id == 4:
            # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0, 0.0, 0.0, 0.0]
        elif camera_model_id == 5:
            # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0, 0.0, 0.0, 0.0]
        elif camera_model_id == 6:
            # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif camera_model_id == 7:
            # FOV: fx, fy, cx, cy, omega
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0]
        elif camera_model_id == 8:
            # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k1
            camera_params = [max(width, height), width / 2, height / 2, 0.0]
        elif camera_model_id == 9:
            # RADIAL_FISHEYE: f, cx, cy, k1, k2
            camera_params = [max(width, height), width / 2, height / 2, 0.0, 0.0]
        elif camera_model_id == 10:
            # THIN_PRISM_FISHEYE: "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1";
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif camera_model_id == 11:
            # RAD_TAN_THIN_PRISM_FISHEYE: "fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3"
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            logger.warning(f"Unsupported camera model ID {camera_model_id}, defaulting to OPENCV parameters")
            camera_params = [max(width, height), max(width, height), width / 2, height / 2, 0.0, 0.0, 0.0, 0.0]
            camera_model_id = 4  # OPENCV
        
        camera_id = db.add_camera(
            model=camera_model_id,
            width=width,
            height=height,
            params=camera_params,
            prior_focal_length=False
        )
        logger.info(f"Added camera: model={camera_model}, id={camera_id}")
        db.commit()
        
        # Add images to database
        image_count = 0
        for image_idx, image_file in enumerate(image_files):
            try:
                image_id = db.add_image(
                    name=image_file,
                    camera_id=camera_id
                )
                image_count += 1
                if (image_idx + 1) % 100 == 0:
                    logger.info(f"  Added {image_idx + 1}/{len(image_files)} images to database")
            except Exception as e:
                logger.warning(f"Failed to add image {image_file}: {e}")
                continue
        
        db.commit()
        db.close()
        
        logger.info(f"Successfully initialized database with {image_count} images")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


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

input_img_num = count_images_in_dir_recursive(images_path)

# --- Feature extraction ---
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
    # "--FeatureExtraction.max_image_size", str(args.max_image_size),
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
    "--AlikedExtraction.max_num_features", "2048",
    "--AlikedExtraction.min_score", "0.2",
    # "--AlikedExtraction.n16rot_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx;aliked-n16rot.onnx;39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    # "--AlikedExtraction.n32_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n32.onnx;aliked-n32.onnx;a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c",

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
        "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT, ALIKED_LIGHTGLUE, ALIKED_N32
        # "--FeatureMatching.num_threads", str(args.num_threads),
        "--FeatureMatching.use_gpu", str(use_gpu),
        # "--FeatureMatching.gpu_index", str(args.gpu_index),
        "--FeatureMatching.guided_matching", "0",
        # "--FeatureMatching.skip_geometric_verification", str(args.skip_geometric_verification),
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", "32768",
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.cpu_brute_force_matcher", "0",
        "--SiftMatching.lightglue_min_score", "0.1",
        # "--SiftMatching.lightglue_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx;sift-lightglue.onnx;e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
        "--AlikedMatching.brute_force_min_cossim", "0.85",
        "--AlikedMatching.brute_force_max_ratio", "1",
        "--AlikedMatching.brute_force_cross_check", "1",
        # "--AlikedMatching.bruteforce_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/bruteforce-matcher.onnx;bruteforce-matcher.onnx;3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47"
        "--AlikedMatching.lightglue_min_score", "0.1",
        # "--AlikedMatching.lightglue_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx;aliked-lightglue.onnx;b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
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
        "--match_type", "pairs", # {'pairs', 'raw', 'inliers'}               
        "--FeatureMatching.type", feature_match_type, # UNDEFINED, SIFT, ALIKED_LIGHTGLUE, ALIKED_N32
        "--FeatureMatching.num_threads", "-1",
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.guided_matching", "0",
        "--FeatureMatching.skip_geometric_verification", "0"
        "--FeatureMatching.rig_verification", "0",
        "--FeatureMatching.skip_image_pairs_in_same_frame", "0",
        "--FeatureMatching.max_num_matches", "32768",
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.cpu_brute_force_matcher", "0",
        "--SiftMatching.lightglue_min_score", "0.1",
        # "--SiftMatching.lightglue_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx;sift-lightglue.onnx;e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
        "--AlikedMatching.brute_force_min_cossim", "0.85",
        "--AlikedMatching.brute_force_max_ratio", "1",
        "--AlikedMatching.brute_force_cross_check", "1",
        # "--AlikedMatching.bruteforce_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/bruteforce-matcher.onnx;bruteforce-matcher.onnx;3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
        "--AlikedMatching.lightglue_min_score", "0.1",
        # "--AlikedMatching.lightglue_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx;aliked-lightglue.onnx;b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
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
        "--FeatureMatching.max_num_matches", "32768",
        "--SiftMatching.max_ratio", "0.8",
        "--SiftMatching.max_distance", "0.7",
        "--SiftMatching.cross_check", "1",
        "--SiftMatching.cpu_brute_force_matcher", "0",
        "--SiftMatching.lightglue_min_score", "0.1",
        # "--SiftMatching.lightglue_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx;sift-lightglue.onnx;e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
        "--AlikedMatching.brute_force_min_cossim", "0.85",
        "--AlikedMatching.brute_force_max_ratio", "1",
        "--AlikedMatching.brute_force_cross_check", "1",
        # "--AlikedMatching.bruteforce_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/bruteforce-matcher.onnx;bruteforce-matcher.onnx;3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
        "--AlikedMatching.lightglue_min_score", "0.1",
        # "--AlikedMatching.lightglue_model_path", "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx;aliked-lightglue.onnx;b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
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
        "--VocabTreeMatching.num_images", "20", # 100
        "--VocabTreeMatching.num_nearest_neighbors", "1", # 5
        "--VocabTreeMatching.num_checks", "32",
        "--VocabTreeMatching.num_images_after_verification", "0", # 0
        "--VocabTreeMatching.max_num_features", "-1",
        # "--VocabTreeMatching.vocab_tree_path", vocab_path,
        # "--VocabTreeMatching.match_list_path", match_list_path,
        "--VocabTreeMatching.num_threads", "-1"
    ]

run_subprocess(feat_matching_cmd, logger)
feature_matching_time = time.time() - t1
logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

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
if args.alg == "acc":
    mapper_cmd = [
        colmap_command, "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.num_threads", "-1",
        "--Mapper.min_num_matches", "30", # 15
        "--Mapper.init_num_trials", "500", # 200
        "--Mapper.init_min_num_inliers", "200", # 100
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
        "--Mapper.ba_global_max_num_iterations", "20", # 50 --> 20 --> 25
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