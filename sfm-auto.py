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
import ctypes

# --------------------------
# Argument parser
# --------------------------
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--source_path", "-s", default="E:\\Test1234\\data18", type=str)
parser.add_argument("--output_path", "-o", default="output_splg", type=str)
parser.add_argument("--camera", default="SIMPLE_RADIAL", type=str)
parser.add_argument("--default_focal_length_factor", default=1.0, type=float, help="Default focal length as a factor of image size (if not specified in EXIF)")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--single_camera", "-sc",default="0", type=str)
parser.add_argument("--single_fold", "-sf", default="1", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--feature_type", type=str, default="SIFT", choices=["SIFT", "ALIKED_N16ROT", "ALIKED_N32"], help="Feature type for COLMAP feature extraction (e.g., SIFT, ALIKED_N16ROT, ALIKED_N32)")
parser.add_argument("--match_strategy", "-ms", type=str, default="vocab_tree", choices=["exhaustive", "sequential", "vocab_tree"], help="Matching strategy to use")
parser.add_argument("--match_type", "-mt", type=str, default="LIGHTGLUE", choices=["BRUTEFORCE", "LIGHTGLUE"], help="Matching type for COLMAP (e.g., ALIKED_LIGHTGLUE, ALIKED_N32)")
parser.add_argument("--mapper", default="global", type=str, choices=["global", "hierarchical", "hierarchical_acc"], help="Algorithm for matching and mapping: colmap / acc / global / hierarchical / hierarchical_acc")
# parser.add_argument("--max_feature_num", "-mfn", default=2048, type=int, help="Maximum number of features to extract per image (for SuperPoint)")
# parser.add_argument("--max_matches_per_image", "-mpi", type=int, default=50,
#                     help="Max number of similar images to match per image (for nearest_k/quick strategies)")
# parser.add_argument("--similarity_threshold", "-st", type=float, default=0.75,
#                     help="Similarity threshold for threshold-based matching strategy (0~1)")
# parser.add_argument("--min_num_inliers", type=int, default=30, help="Minimum number of inliers for a valid match")
# parser.add_argument("--min_inlier_ratio", type=float, default=0.1, help="Minimum inlier ratio for a valid match")
# parser.add_argument("--sequential_overlap", "-so", type=int, default=15, help="Number of neighboring images to match on each side for sequential matching")
# parser.add_argument("--filt_match", action="store_true", help="Whether to filter matches by inliers before mapping")
# parser.add_argument("--filter_inlier_ratio_threshold", type=float, default=0.6, help="Inlier ratio threshold for filtering matches before mapping")
# parser.add_argument("--filter_inlier_num_threshold", type=int, default=30, help="Inlier number threshold for filtering matches before mapping")
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
    if not Path(output_path).is_absolute():
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

bruteforce_match_path = os.path.join(current_path, "checkpoints/colmap_dep/bruteforce_matcher.onnx")
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

vocab_path = vocab_sift_path
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

colmap_log_path = os.path.join(output_path, "colmap.log")

# ======================== GPU setup ==========================================
use_gpu = 0 if args.no_gpu else 1

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
    # sys.exit(1)

logger.info(f"Found {len(image_files)} images")

# =============================================================================
auto_reconstruct_cmd = [
    colmap_command, "automatic_reconstructor",
    # "--default_random_seed", "0",
    "--log_target", "stderr_and_file",
    "--log_path", str(colmap_log_path),
    "--log_level", str(log_level),
    "--log_severity", "0",
    "--log_color", "1",
    "--workspace_path", output_path,
    "--image_path", images_path,
    "--vocab_tree_path", vocab_path,
    "--data_type", "individual",
    "--quality", "EXTREME",
    "--camera_model", args.camera,
    "--single_camera", str(args.single_camera),
    "--single_camera_per_folder", str(args.single_fold),
    # "--camera_params", "",
    "--extraction", "1",
    "--matching", "1",
    "--sparse", "1",
    "--dense", "0",
    "--feature", args.feature_type,
    "--mapper", "GLOBAL",
    # "--mesher", "0",
    "--num_threads", "-1",
    "--random_seed", "-1",
    "--use_gpu", str(use_gpu),
    "--gpu_index", "-1",
    ]

logger.info("Starting auto reconstruct with COLMAP SIFT...")
t0 = time.time()
run_subprocess(auto_reconstruct_cmd, logger)
feature_extraction_time = time.time() - t0
logger.info(f"Auto reconstruct done in {feature_extraction_time/60:.2f} min")