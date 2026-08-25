import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from mapper_cmd import (
    get_ba_cmd,
    get_incremental_mapper_cmd,
    get_hierarchical_mapper_cmd,
    get_global_mapper_cmd,
    get_points_triangulate_cmd,
    get_pose_prior_global_mapper_cmd,
    get_pos_prior_mapper_cmd,
    get_reconstruction_refine_cmd,
    get_model_align_cmd
)
from record_param import record_args
from utils import (
    delete_directory,
    run_subprocess,
    check_operating_system,
    count_images_in_dir_recursive,
    get_largest_subfolder,
    get_subfolders_names,
    get_subfolders,
    clear_folder,
    move_files,
)
from database import ReadColmapDatabase, filter_matches_by_inliers, read_all_keypoints, get_matched_image_pairs
from visualization import visualize_image_pairs, draw_keypoints_on_image
from calibration_utils import build_prior_cameras_from_calibration

from match_utils import compute_matched_image_pairs_by_pose_prior
from database import initialize_colmap_database, initialize_colmap_database_from_prior_camera_file
from feature_extractor_cmd import get_feature_extractor_cmd
from match_cmd import (
    get_exhaustive_matcher_cmd,
    get_spatial_matcher_cmd,
    get_matches_importer_cmd,
    get_sequential_match_list,
    get_vocab_tree_matcher_cmd
)
from read_write_model import read_images_binary, write_images_binary

#  ========================== Argument parser ==========================
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--ba_local_backend", default="CERES", type=str, choices=["CERES", "CASPAR"], help="Local BA backend (e.g., Ceres, g2o)")
parser.add_argument("--ba_global_backend", default="CERES", type=str, choices=["CERES", "CASPAR"], help="Global BA backend (e.g., Ceres, g2o)")
parser.add_argument("--source_path", "-s", default="E:\\debug", type=str)
parser.add_argument("--pos_file", "-pf", default="", type=str, help="Path to gps or cartesian pose file (if available)")
parser.add_argument("--output_path", "-o", default="output-debug", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--default_focal_length_factor", default=1.2, type=float, help="Default focal length as a factor of image size (if not specified in EXIF)")
parser.add_argument("--camera_params", default="", type=str, help="Camera parameters for COLMAP")
parser.add_argument("--prior_camera_file", "-pcf", default="", type=str, help="Path to prior camera intrinsics file (one camera per line: CAMERA_ID, FOLD, MODEL, WIDTH, HEIGHT, PARAMS). When provided, the database is initialized from this file.")
parser.add_argument("--refine_focal_length", type=int, default=1, help="Whether to refine focal length during bundle adjustment")
parser.add_argument("--refine_principal_point", type=int, default=1, help="Whether to refine principal point during bundle adjustment")
parser.add_argument("--refine_extra_params", type=int, default=1, help="Whether to refine extra camera parameters during bundle adjustment")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--init_camera", action="store_true")
parser.add_argument("--single_camera", "-sc",default="0", type=str)
parser.add_argument("--single_fold", "-sf", default="1", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--feature_type", "-ft", type=str, default="LOMA_B128", choices=["SIFT", "ALIKED_N16ROT", "ALIKED_N32", "LOMA_B", "LOMA_B128"], help="Feature type for COLMAP feature extraction (e.g., SIFT, ALIKED_N16ROT, ALIKED_N32)")
parser.add_argument("--max_image_size", type=int, default=-1, help="maximum image size used to extract feature")
parser.add_argument("--match_strategy", "-ms", type=str, default="vocab_tree", choices=["exhaustive", "sequential", "vocab_tree", "spatial", "threshold", "custom"], help="Matching strategy to use")
parser.add_argument("--match_alg", "-ma", type=str, default="LOMA_B128", choices=["SIFT_BRUTEFORCE", "ALIKED_BRUTEFORCE", "LOMA_BRUTEFORCE", "SIFT_LIGHTGLUE", "ALIKED_LIGHTGLUE", "LOMA_B", "LOMA_B128", "LOMA_R", "LOMA_L", "LOMA_G"], help="Matching type for COLMAP (e.g., ALIKED_LIGHTGLUE, ALIKED_N32)")
parser.add_argument("--vocab_feature_num", type=int, default=0, help="vocab tree retrial feature num")
parser.add_argument("--mapper", default="global", type=str, choices=["incremental", "acc", "global", "hierarchical", "hierarchical_acc", "pos_prior", "pose_prior_global", "pose_prior_incremental"], help="Algorithm for matching and mapping: colmap / acc / global / hierarchical / hierarchical_acc / pose_prior")
parser.add_argument("--max_feature_num", "-mfn", default=2000, type=int, help="Maximum number of features to extract per image")
parser.add_argument("--anms_selected_num", "-asn", default=-1, type=int, help="Maximum number of features to retain per image after final selection")
parser.add_argument("--cell_num", "-cn", default=-1, type=int, help="Number of cells for ANMS feature selection")
parser.add_argument("--per_cell_num", "-pcn", default=-1, type=int, help="Number of features to retain per cell for ANMS feature selection")
parser.add_argument("--sift_peak_threshold", "-spt", default=0.02, type=float, help="SIFT peak threshold for feature extraction")
parser.add_argument("--sift_first_octave", "-sfo", default=0, type=int, help="SIFT first octave for feature extraction")
parser.add_argument("--sift_match_max_distance", "-smmd", default=0.7, type=float, help="SIFT match max distance for feature matching")
parser.add_argument("--sift_match_max_ratio", "-smmr", default=0.7, type=float, help="SIFT match max ratio for feature matching")
parser.add_argument("--loma_extractor_min_score", type=float, default=0.0, help="Minimum score threshold for LOMA feature extraction")
parser.add_argument("--loma_extractor_use_bf16", type=int, default=0, help="Whether to use BF16 for LOMA feature extraction")
parser.add_argument("--loma_extractor_use_fast_resize", type=int, default=0, help="Whether to use fast resize for LOMA feature extraction")
parser.add_argument("--loma_match_min_score", type=float, default=0.1, help="Minimum score threshold for LOMA feature matching")
parser.add_argument("--loma_match_use_bf16", type=int, default=0, help="Whether to use BF16 for LOMA feature matching")
parser.add_argument("--min_num_inliers", type=int, default=15, help="Minimum number of inliers for a valid match")
parser.add_argument("--min_inlier_ratio", type=float, default=0.1, help="Minimum inlier ratio for a valid match")
parser.add_argument("--sequential_overlap", "-so", type=int, default=15, help="Number of neighboring images to match on each side for sequential matching")
parser.add_argument("--two_view_geometry_max_error", "-tvgme", type=float, default=4.0, help="two viw geometry max error")
parser.add_argument("--filt_match", type=int, default=0, help="Whether to filter matches by inliers before mapping")
parser.add_argument("--filter_inlier_ratio_threshold", type=float, default=0.2, help="Inlier ratio threshold for filtering matches before mapping")
parser.add_argument("--filter_inlier_num_threshold", type=int, default=15, help="Inlier number threshold for filtering matches before mapping")
parser.add_argument("--ra_max_rotation_error_deg", type=float, default=10.0, help="Maximum rotation error in degrees for rotation averaging")
parser.add_argument("--ra_max_rotation_error_final_deg", type=float, default=10.0, help="Maximum rotation error in degrees for final rotation averaging")
parser.add_argument("--ra_refilt_outlier_pairs_num", type=int, default=10, help="Number of outlier pairs to re-filter in rotation averaging")
parser.add_argument("--gp_max_num_iterations", type=int, default=200, help="Maximum number of iterations for global positioning")
parser.add_argument("--ba_ceres_max_num_iterations", type=int, default=200, help="Maximum number of iterations for Ceres bundle adjustment")
parser.add_argument("--max_normalized_reproj_error", type=float, default=0.01, help="Maximum normalized reprojection error")
parser.add_argument("--global_mapper_min_tri_angle_deg", type=float, default=1.0, help="Minimum triangulation angle in degrees for global mapper")
parser.add_argument("--track_min_num_views_per_track", type=int, default=4, help="Minimum number of views per track")
parser.add_argument("--final_min_num_points3d", type=int, default=3, help="Final minimum number of 3D points")
parser.add_argument("--final_min_num_covisible_images", type=int, default=3, help="Final minimum number of covisible images")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
parser.add_argument("--visualize_matches", "-vis", action="store_true", help="Whether to visualize matches")
parser.add_argument("--visualize_keypoints", "-viskpts", action="store_true", help="Whether to visualize all keypoints on each image")
parser.add_argument("--clean", action="store_true", help="Whether to clean the output directory")
parser.add_argument("--farest_image_distance", "-fid", type=float, default=400.0, help="Maximum distance between images for spatial matching")
parser.add_argument("--max_matches_per_image", "-mpi", type=int, default=50,
                    help="Max number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--min_matches_per_image", "-mni", type=int, default=50,
                    help="Minimum number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--similarity_threshold", "-st", type=float, default=0.75,
                    help="Similarity threshold for threshold-based matching strategy (0~1)")
parser.add_argument("--pose_prior", type=str, default="", help="Path to prior pose file (if available)")
parser.add_argument("--voxel_size", type=float, default=None, help="Voxel size for prior pose-based image pair generation")
parser.add_argument("--max_angle", type=float, default=120, help="Maximum angle (in degrees) between camera views for prior pose-based image pair generation")
parser.add_argument("--min_overlap", type=float, default=0.1, help="Minimum frustum overlap for prior pose-based image pair generation")
parser.add_argument("--refine_num", type=int, default=1, help="refine reconstruction iterations num")
parser.add_argument("--undistort", type=int, default=0, help="Whether to undistort images after reconstruction")
parser.add_argument("--unify_output_images", action="store_true")
parser.add_argument("--monitor_memory", action="store_true", help="Monitor and report peak CPU RAM and GPU VRAM usage of COLMAP processes")
parser.add_argument("--create_mask", action="store_true", help="Create binary image masks before feature extraction")
parser.add_argument("--corner_width", type=float, default=0.0, help="Width ratio of each corner mask region, in [0, 1]")
parser.add_argument("--corner_height", type=float, default=0.0, help="Height ratio of each corner mask region, in [0, 1]")
parser.add_argument("--center_width", type=float, default=0.0, help="Width ratio of the center mask region, in [0, 1]")
parser.add_argument("--center_height", type=float, default=0.0, help="Height ratio of the center mask region, in [0, 1]")
args = parser.parse_args()


# ============================== output path ============================
if args.output_path == "":
    output_path = args.source_path
else:
    output_path = args.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.source_path, output_path)
os.makedirs(output_path, exist_ok=True)


# ========================== Helper functions =========================
def resource_path() -> str:
    """Get absolute path for packaged or development scripts."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path


def create_image_masks(
    images_dir: str,
    image_paths: list,
    mask_dir: str,
    corner_width: float,
    corner_height: float,
    center_width: float,
    center_height: float,
) -> str:
    """Create binary masks while preserving the input image relative paths."""
    mask_sizes = (corner_width, corner_height, center_width, center_height)
    if any(value < 0 or value > 1 for value in mask_sizes):
        raise ValueError("Mask dimension ratios must be between 0 and 1")

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except (OSError, ValueError) as error:
            raise ValueError(f"Failed to read image header for mask creation: {image_path}") from error

        corner_w = int(round(corner_width * image_width))
        corner_h = int(round(corner_height * image_height))
        center_w = int(round(center_width * image_width))
        center_h = int(round(center_height * image_height))

        mask = np.full((image_height, image_width), 255, dtype=np.uint8)

        # Mask the four corner rectangles.
        mask[:corner_h, :corner_w] = 0
        mask[:corner_h, image_width - corner_w:] = 0
        mask[image_height - corner_h:, :corner_w] = 0
        mask[image_height - corner_h:, image_width - corner_w:] = 0

        # Mask the centered rectangle.
        center_x0 = max((image_width - center_w) // 2, 0)
        center_y0 = max((image_height - center_h) // 2, 0)
        mask[center_y0:center_y0 + center_h, center_x0:center_x0 + center_w] = 0

        relative_path = os.path.relpath(image_path, images_dir)
        mask_path = os.path.join(mask_dir, relative_path + ".png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        print(f"Creating mask for {image_path} -> {mask_path}")

        # Store the lossless mask as <original filename>.png.
        encoded_mask = cv2.imencode(".png", mask)[1]
        encoded_mask.tofile(mask_path)

    return mask_dir


def create_mask_for_image(
    image_path: str,
    mask_path: str,
    corner_width: float,
    corner_height: float,
    center_width: float,
    center_height: float,
) -> str:
    """Create one binary mask with the same dimensions as an input image."""
    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size
    except (OSError, ValueError) as error:
        raise ValueError(f"Failed to read image header for mask creation: {image_path}") from error

    mask_sizes = (corner_width, corner_height, center_width, center_height)
    if any(value < 0 or value > 1 for value in mask_sizes):
        raise ValueError("Mask dimension ratios must be between 0 and 1")

    corner_w = int(round(corner_width * image_width))
    corner_h = int(round(corner_height * image_height))
    center_w = int(round(center_width * image_width))
    center_h = int(round(center_height * image_height))
    mask = np.full((image_height, image_width), 255, dtype=np.uint8)
    mask[:corner_h, :corner_w] = 0
    mask[:corner_h, image_width - corner_w:] = 0
    mask[image_height - corner_h:, :corner_w] = 0
    mask[image_height - corner_h:, image_width - corner_w:] = 0

    center_x0 = max((image_width - center_w) // 2, 0)
    center_y0 = max((image_height - center_h) // 2, 0)
    mask[center_y0:center_y0 + center_h, center_x0:center_x0 + center_w] = 0

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    encoded_mask = cv2.imencode(".png", mask)[1]
    encoded_mask.tofile(mask_path)
    return mask_path

# ============================ Logging setup ===============================
log_level = args.log_level
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger()
logger.setLevel(log_level)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(log_level)
ch.setFormatter(log_formatter)
logger.addHandler(ch)

# File handler
log_dir = output_path
os.makedirs(log_dir, exist_ok=True)
now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = os.path.join(log_dir, f"run-sfm-{now_str}.log")
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(log_level)
fh.setFormatter(log_formatter)
logger.addHandler(fh)

# =========================== Record parameters to log file ==========================
# record_args(logger, args)

# ============================ Timeing variables ===============================
start_time = time.time()
pre_match_time = 0
feature_extraction_time = 0
feature_matching_time = 0
view_graph_calibrate_time = 0
triangulate_time = 0
ba_time = 0
refine_time = 0
mapper_time = 0

# ===================== Memory monitoring accumulator ============================
peak_memory: dict = {}  # populated by run_subprocess when --monitor_memory is set

# =====================key parameter =====================================================
min_num_inliers = args.min_num_inliers
min_inlier_ratio = args.min_inlier_ratio

feature_match_type = args.match_alg

# ===================== Paths and executables ============================================
os_type = check_operating_system()
current_path = resource_path()
print(f"Detected operating system: {os_type}")
if os_type == 'Windows':
    # colmap_path = os.path.join(current_path, "colmap-x64-windows-cuda-4.0.4/bin/colmap.exe")
    # colmap_path = os.path.join(current_path, "Release-colmap-ch/colmap.exe")
    colmap_path = "D:\\Codes\\Study\\colmap\\build\\src\\colmap\\exe\\Release\\colmap.exe"
    # colmap_path = "D:\\Programs\\colmap-x64-windows-cuda-4.1.0\\bin\\colmap.exe"
    # colmap_path = os.path.join(current_path, "Release-colmap-4.2.0-dev-wl-260819/colmap.exe")
    
else:
    colmap_path = "colmap"

colmap_command = args.colmap_executable if args.colmap_executable else colmap_path

bruteforce_match_path = os.path.join(current_path, "checkpoints/colmap_dep/bruteforce-matcher.onnx")
sift_lightglue_match_path = os.path.join(current_path, "checkpoints/colmap_dep/sift-lightglue.onnx")
aliked_lightglue_match_path = os.path.join(current_path, "checkpoints/colmap_dep/aliked-lightglue.onnx")
aliked_n16rot_path = os.path.join(current_path, "checkpoints/colmap_dep/aliked-n16rot.onnx")
aliked_n32_path = os.path.join(current_path, "checkpoints/colmap_dep/aliked-n32.onnx")
loma_detector_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_detector.onnx")
loma_descriptor_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_descriptor_dedode_g.onnx")
loma_descriptor_model_path_bf16 = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_descriptor_dedode_g_bf16.onnx")
loma_descriptor_b128_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_descriptor_dedode_b.onnx")
loma_match_b_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_B.onnx")
loma_match_b_model_path_bf16 = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_B_bf16.onnx")
loma_match_b128_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_B128.onnx")
loma_match_b128_model_path_bf16 = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_B128_bf16.onnx")
loma_match_r_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_R.onnx")
loma_match_r_model_path_bf16 = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_R_bf16.onnx")
loma_match_l_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_L.onnx")
loma_match_l_model_path_bf16 = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_L_bf16.onnx")
loma_match_g_model_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_G.onnx")
loma_match_g_model_path_bf16 = os.path.join(current_path, "checkpoints/colmap_dep/loma/loma_matcher_G_bf16.onnx")
loma_match_brute_force_min_cossim = 0.85
loma_match_brute_force_max_ratio = 1
loma_match_brute_force_cross_check = 1
loma_match_brute_force_model_path = bruteforce_match_path

vocab_sift_path = os.path.join(current_path, "checkpoints/colmap_dep/vocab_tree_faiss_flickr100K_words256K.bin")
vocab_aliked_n16rot_path = os.path.join(current_path, "checkpoints/colmap_dep/vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin")
vocab_aliked_n32_path = os.path.join(current_path, "checkpoints/colmap_dep/vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin")
vocab_loma_b_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/vocab_tree_loma_b_flickr100k.bin")
vocab_loma_b128_path = os.path.join(current_path, "checkpoints/colmap_dep/loma/vocab_tree_loma_b128_flickr100k.bin")

if args.feature_type == "SIFT":
    vocab_path = vocab_sift_path
elif args.feature_type == "ALIKED_N16ROT":
    vocab_path = vocab_aliked_n16rot_path
elif args.feature_type == "ALIKED_N32":
    vocab_path = vocab_aliked_n32_path
elif args.feature_type == "LOMA_B":
    vocab_path = vocab_loma_b_path
elif args.feature_type == "LOMA_B128":
    vocab_path = vocab_loma_b128_path
else:
    vocab_path = ""

# ========================== clean output directory ==========================
if args.clean:
    logger.info(f"Cleaning output directory: {output_path}")
    try:
        clear_folder(output_path)
    except Exception as e:
        logger.info(f"Failed to clean output directory: {e}")   

# ========================== Start of the pipeline ============================
logger.info("=== Starting Structure-from-Motion Pipeline ===")
logger.info(f"Source path: {args.source_path}")
logger.info(f"Output path: {output_path}")
distorted_sparse_path = os.path.join(output_path, "distorted/sparse")
os.makedirs(distorted_sparse_path, exist_ok=True)

database_path = os.path.join(output_path, "distorted/database.db")
images_dir = os.path.join(args.source_path, "input")
matched_images_pairs_path = os.path.join(distorted_sparse_path, "image_pairs.txt")

input_img_num, images_full_path = count_images_in_dir_recursive(images_dir)
logger.info(f"input images num: {input_img_num}")
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
images_full_path = sorted(images_full_path)
logger.info(f"Found {len(images_full_path)} images")

# ======================== GPU setup ==========================================
use_gpu = 0 if args.no_gpu else 1

# ============================ find matched pairs using slam pose (optional) ==========
prior_cameras = None
prior_images = None
if args.pose_prior is not None and len(args.pose_prior) > 0:
    pre_match_start = time.time()
    logger.info(f"Finding image pairs based on low-pricision poses from: {args.pose_prior}")

    if not os.path.isabs(args.pose_prior):
        args.pose_prior = os.path.join(args.source_path, args.pose_prior)
    prior_cameras, prior_images = compute_matched_image_pairs_by_pose_prior(
        args.pose_prior,
        matched_images_pairs_path,
        voxel_size=args.voxel_size,
        max_angle=args.max_angle,
        min_overlap=args.min_overlap,
    )

    # prior_images 为 (image_id, image_path, camera_id) 元组列表，
    # 此处仅提取路径用于图像过滤
    prior_images_list = [item[1] for item in prior_images]
    first_cam = next(iter(prior_cameras.values()))
    prior_focal_length = first_cam.params[0]
    prior_focal_factor = prior_focal_length / max(first_cam.width, first_cam.height)
    logger.info(
        f"prior-pose-based image pairs saved to: {matched_images_pairs_path}, prior focal length: {prior_focal_length}, prior focal factor: {prior_focal_factor}"
    )

    args.default_focal_length_factor = prior_focal_factor

    # 统一路径分隔符为 "/"
    prior_images_set = {name.replace('\\', '/') for name in prior_images_list}

    # 遍历副本，避免在遍历过程中原地删除导致漏判/漏删
    for img_path in list(images_full_path):
        rel_path = os.path.relpath(img_path, images_dir).replace('\\', '/')
        if rel_path not in prior_images_set:
            logger.warning(f"Image {rel_path} does not have prior-pose, please check consistency between prior pose file and input images")
            images_full_path.remove(img_path)

    # input("Press Enter to continue with feature extraction and mapping using the prior-pose-based image pairs...")
    pre_match_time = time.time() - pre_match_start
    logger.info(f"Using {len(images_full_path)} images, pre-matching time: {pre_match_time:.2f} s")


# ======================== Initialize database ==================================
# 若指定了先验相机内参文件（--prior_camera_file），则优先使用该文件初始化数据库：
# 每个子文件夹对应一个相机，相机 ID 与内参均取自文件。
prior_camera_file = args.prior_camera_file
if prior_camera_file:
    if not os.path.isabs(prior_camera_file):
        prior_camera_file = os.path.join(args.source_path, prior_camera_file)
    if not os.path.isfile(prior_camera_file):
        logger.error(f"Prior camera file not found: {prior_camera_file}")
        sys.exit(1)

    logger.info("Initializing COLMAP database from prior camera file ...")
    db_init_success = initialize_colmap_database_from_prior_camera_file(
        database_path=database_path,
        images_dir=images_dir,
        prior_camera_file=prior_camera_file,
        logger=logger,
    )
    if not db_init_success:
        logger.error("Failed to initialize database from prior camera file!")
        sys.exit(1)

# 从 calibration.json 读取相机内参先验，构建为与 prior_cameras 相同的数据结构
# 约定：left → camera 1, right → camera 2
else:
    if prior_cameras is None:
        calib_result = build_prior_cameras_from_calibration(
            calib_path=os.path.join(args.source_path, "calibration.json"),
            images_dir=images_dir,
            images_full_path=images_full_path,
            logger=logger,
        )
        if calib_result is not None:
            prior_cameras, prior_images = calib_result
            args.init_camera = True

    if args.init_camera or prior_cameras:
        camera_assignment = "global" 
        if args.single_camera == "1": 
            camera_assignment = "global"
        elif args.single_fold == "1":
            camera_assignment = "per_subfolder"
        else:
            camera_assignment = "per_image"

        logger.info("Initializing COLMAP database with image metadata ...")
        db_init_success = initialize_colmap_database(
            database_path=database_path,
            images_dir= images_dir,
            input_images_path=images_full_path,
            camera_model=args.camera,
            camera_assignment=camera_assignment,
            prior_cameras=prior_cameras,
            prior_images=prior_images,
            logger=logger
        )    
        if not db_init_success:
            logger.error("Failed to initialize database!")
            sys.exit(1)


# ========================= Image mask generation =============================
camera_mask_path = ""
image_mask_path = ""
if args.create_mask:
    input_subfolders = [
        path for path in get_subfolders(images_dir)
        if os.path.basename(path) != "mask"
    ]
    mask_params = (
        args.corner_width,
        args.corner_height,
        args.center_width,
        args.center_height,
    )

    if len(input_subfolders) <= 1:
        if not images_full_path:
            logger.error("Cannot create mask: no input images found")
            sys.exit(1)
        camera_mask_path = os.path.join(args.source_path, "mask.png")
        create_mask_for_image(images_full_path[0], camera_mask_path, *mask_params)
        logger.info(f"Created camera mask: {camera_mask_path}")
    else:
        image_mask_path = os.path.join(args.source_path, "mask")
        create_image_masks(
            images_dir=images_dir,
            image_paths=images_full_path,
            mask_dir=image_mask_path,
            corner_width=args.corner_width,
            corner_height=args.corner_height,
            center_width=args.center_width,
            center_height=args.center_height,
        )
        logger.info(f"Created image masks under: {image_mask_path}")


# ========================= Feature extraction ==================================
feat_extraction_cmd = get_feature_extractor_cmd(
    colmap_command=colmap_command,
    log_level=log_level,
    database_path=database_path,
    images_path=images_dir,
    feature_type=args.feature_type,
    camera_mask_path=camera_mask_path,
    image_mask_path=image_mask_path,
    single_camera_per_image=int(args.single_image),
    single_camera_per_fold=int(args.single_fold),
    single_camera=int(args.single_camera),
    camera_model=args.camera,
    default_focal_length_factor=args.default_focal_length_factor,
    camera_parameters=args.camera_params,
    use_gpu=use_gpu,
    max_image_size=args.max_image_size,
    max_feature_num=args.max_feature_num,
    anms_selected_num=args.anms_selected_num,
    cell_num=args.cell_num,
    per_cell_num=args.per_cell_num,
    sift_peak_threshold=args.sift_peak_threshold,
    sift_first_octave=args.sift_first_octave,
    aliked_n16rot_path=aliked_n16rot_path,
    aliked_n32_path=aliked_n32_path,
    loma_detector_model_path=loma_detector_model_path,
    loma_descriptor_model_path=loma_descriptor_model_path,
    loma_descriptor_model_path_bf16=loma_descriptor_model_path_bf16,
    loma_descriptor_b128_model_path=loma_descriptor_b128_model_path)

logger.info("Starting feature extraction with COLMAP ...")
t0 = time.time()
run_subprocess(feat_extraction_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
feature_extraction_time = time.time() - t0
logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

if args.pos_file:
    pose_importer_cmd = [
        colmap_command, "pose_prior_importer",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--pose_prior_json_path", args.pos_file,
    ]
    run_subprocess(pose_importer_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)

# ======================== config Rig =============================================
# input("Press Enter to continue with feature matching and mapping...")
config_rig_cmd = [
    colmap_command, "rig_configurator",
    "--log_level", str(log_level),
    "--database_path", database_path,
    "--rig_config_path", "E:\\M3D_Test_Data\\town\\input\\rig_config.json",
    "--input_path", "E:\\M3D_Test_Data\\town\\output-sift_0.03_8000_2000-vocab_150_0.7_0.7_filt_0.3_15-global\\distorted\\sparse\\0", # 0.5
]
# run_subprocess(config_rig_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
# input("Press Enter to continue with feature matching and mapping...")
# ========================= Visualize keypoints (optional) =========================
if args.visualize_keypoints:
    logger.info("Visualizing all keypoints on images...")
    keypoints_dict, image_names = read_all_keypoints(database_path)
    if not keypoints_dict:
        logger.warning("No keypoints found in database, skipping keypoint visualization")
    else:
        vis_kpts_dir = os.path.join(output_path, 'keypoints_vis')
        os.makedirs(vis_kpts_dir, exist_ok=True)
        for img_id, kpts in keypoints_dict.items():
            img_name = image_names[img_id]
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}, skipping")
                continue
            # 输出路径使用相对路径名避免嵌套目录
            safe_name = img_name.replace('/', '_').replace('\\', '_')
            kp_vis_output_path = os.path.join(vis_kpts_dir, f"kpts_{safe_name}")
            draw_keypoints_on_image(img_path, kpts, kp_vis_output_path, max_points=5000, radius=20)
            logger.info(f"  Visualized {len(kpts)} keypoints on {img_name}")
        logger.info(f"Keypoint visualization saved to: {vis_kpts_dir}")


# ========================= Feature matching =========================
logger.info("Starting feature matching...")
 
match_start = time.time()
if args.match_strategy == "exhaustive":
    feat_matching_cmd = get_exhaustive_matcher_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        feature_match_type=feature_match_type,
        use_gpu=use_gpu,
        max_feature_num=args.max_feature_num*4,
        min_num_inliers=min_num_inliers,
        min_inlier_ratio=min_inlier_ratio,
        sift_match_max_distance=args.sift_match_max_distance,
        sift_match_max_ratio=args.sift_match_max_ratio,
        sift_lightglue_match_path=sift_lightglue_match_path,
        bruteforce_match_path=bruteforce_match_path,
        aliked_lightglue_match_path=aliked_lightglue_match_path,
        b_model_path=loma_match_b_model_path,
        b_model_path_bf16=loma_match_b_model_path_bf16,
        b128_model_path=loma_match_b128_model_path,
        b128_model_path_bf16=loma_match_b128_model_path_bf16,
        r_model_path=loma_match_r_model_path,
        r_model_path_bf16=loma_match_r_model_path_bf16,
        l_model_path=loma_match_l_model_path,
        l_model_path_bf16=loma_match_l_model_path_bf16,
        g_model_path=loma_match_g_model_path,
        g_model_path_bf16=loma_match_g_model_path_bf16,          
        two_view_geometry_max_error=args.two_view_geometry_max_error
    )
elif args.match_strategy == "custom":
    logger.info(f"Matching features based on custom image pairs from: {matched_images_pairs_path}")
    
    # Use matches_importer to import the sequential match list
    feat_matching_cmd = get_matches_importer_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        matched_images_pairs_path=matched_images_pairs_path,
        feature_match_type=feature_match_type,
        use_gpu=use_gpu,
        max_feature_num=args.max_feature_num * 4,
        sift_lightglue_match_path=sift_lightglue_match_path,
        bruteforce_match_path=bruteforce_match_path,
        aliked_lightglue_match_path=aliked_lightglue_match_path,
        b_model_path=loma_match_b_model_path,
        b_model_path_bf16=loma_match_b_model_path_bf16,
        b128_model_path=loma_match_b128_model_path,
        b128_model_path_bf16=loma_match_b128_model_path_bf16,
        r_model_path=loma_match_r_model_path,
        r_model_path_bf16=loma_match_r_model_path_bf16,
        l_model_path=loma_match_l_model_path,
        l_model_path_bf16=loma_match_l_model_path_bf16,
        g_model_path=loma_match_g_model_path,
        g_model_path_bf16=loma_match_g_model_path_bf16,          
        min_num_inliers=min_num_inliers,
        min_inlier_ratio=min_inlier_ratio,
        two_view_geometry_max_error=args.two_view_geometry_max_error
    )        

elif args.match_strategy == "sequential":
    # Generate sequential match list with circular overlap
    logger.info("Generating sequential match pairs...")
    
    # Generate match pairs and save to file
    match_pairs = get_sequential_match_list(
        image_names=images_full_path,
        overlap=args.sequential_overlap,
        output_file=matched_images_pairs_path,
        logger=logger
    )
    
    logger.info(f"Generated {len(match_pairs)} sequential match pairs (overlap={args.sequential_overlap})")
    
    # Use matches_importer to import the sequential match list
    feat_matching_cmd = get_matches_importer_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        matched_images_pairs_path=matched_images_pairs_path,
        feature_match_type=feature_match_type,
        use_gpu=use_gpu,
        max_feature_num=args.max_feature_num,
        sift_lightglue_match_path=sift_lightglue_match_path,
        bruteforce_match_path=bruteforce_match_path,
        aliked_lightglue_match_path=aliked_lightglue_match_path,
        b_model_path=loma_match_b_model_path,
        b_model_path_bf16=loma_match_b_model_path_bf16,
        b128_model_path=loma_match_b128_model_path,
        b128_model_path_bf16=loma_match_b128_model_path_bf16,
        r_model_path=loma_match_r_model_path,
        r_model_path_bf16=loma_match_r_model_path_bf16,
        l_model_path=loma_match_l_model_path,
        l_model_path_bf16=loma_match_l_model_path_bf16,
        g_model_path=loma_match_g_model_path,
        g_model_path_bf16=loma_match_g_model_path_bf16,          
        min_num_inliers=min_num_inliers,
        min_inlier_ratio=min_inlier_ratio
    )
elif args.match_strategy == "vocab_tree":
    feat_matching_cmd = get_vocab_tree_matcher_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        feature_match_type=feature_match_type,
        use_gpu=use_gpu,
        vocab_path=vocab_path,  
        sift_lightglue_match_path=sift_lightglue_match_path,
        bruteforce_match_path=bruteforce_match_path,
        aliked_lightglue_match_path=aliked_lightglue_match_path,   
        b_model_path=loma_match_b_model_path,
        b_model_path_bf16=loma_match_b_model_path_bf16,
        b128_model_path=loma_match_b128_model_path,
        b128_model_path_bf16=loma_match_b128_model_path_bf16,
        r_model_path=loma_match_r_model_path,
        r_model_path_bf16=loma_match_r_model_path_bf16,
        l_model_path=loma_match_l_model_path,
        l_model_path_bf16=loma_match_l_model_path_bf16,
        g_model_path=loma_match_g_model_path,
        g_model_path_bf16=loma_match_g_model_path_bf16,                            
        max_feature_num=args.max_feature_num * 4,
        max_matches_per_image=args.max_matches_per_image,
        min_num_inliers=min_num_inliers,
        min_inlier_ratio=min_inlier_ratio,
        sift_match_max_distance=args.sift_match_max_distance,
        sift_match_max_ratio=args.sift_match_max_ratio,
        two_view_geometry_max_error=args.two_view_geometry_max_error,
        vocab_feature_num=0
    )
elif args.match_strategy == "spatial":
    feat_matching_cmd = get_spatial_matcher_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        feature_match_type=feature_match_type,
        use_gpu=use_gpu,
        sift_lightglue_match_path=sift_lightglue_match_path,
        bruteforce_match_path=bruteforce_match_path,
        aliked_lightglue_match_path=aliked_lightglue_match_path,   
        b_model_path=loma_match_b_model_path,
        b_model_path_bf16=loma_match_b_model_path_bf16,
        b128_model_path=loma_match_b128_model_path,
        b128_model_path_bf16=loma_match_b128_model_path_bf16,
        r_model_path=loma_match_r_model_path,
        r_model_path_bf16=loma_match_r_model_path_bf16,
        l_model_path=loma_match_l_model_path,
        l_model_path_bf16=loma_match_l_model_path_bf16,
        g_model_path=loma_match_g_model_path,
        g_model_path_bf16=loma_match_g_model_path_bf16,                           
        max_feature_num=args.max_feature_num * 4,
        min_num_inliers=min_num_inliers,
        min_inlier_ratio=min_inlier_ratio,
        sift_match_max_distance=args.sift_match_max_distance,
        sift_match_max_ratio=args.sift_match_max_ratio,
        two_view_geometry_max_error=args.two_view_geometry_max_error,
        max_num_neighbors=args.max_matches_per_image,
        min_num_neighbors=args.min_matches_per_image,
        farest_image_distance=args.farest_image_distance
    )
else:
    logger.error(f"Unsupported match strategy: {args.match_strategy}")
    sys.exit(1)
run_subprocess(feat_matching_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
feature_matching_time = time.time() - match_start + pre_match_time
logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

matched_image_pairs_num = get_matched_image_pairs(database_path)
logger.info(f"Total matched image pairs: {matched_image_pairs_num}")

# ========================= Filter matches by inliers (optional) =========================
if args.filt_match:
    logger.info("Filtering matches by inliers before mapping...")
    filted_paris_num = filter_matches_by_inliers(
        database_path=database_path,
        min_num_inliers=args.filter_inlier_num_threshold,
        min_inlier_ratio=args.filter_inlier_ratio_threshold,
        logger=logger
    )
    logger.info(f"Filtering done. removed pairs: {filted_paris_num}")

# ========================= Calibrate view graph =========================
view_graph_calibrate_start = time.time()
view_graph_calibrate_cmd = [
    colmap_command, "view_graph_calibrator",
    "--log_level", str(log_level),
    "--database_path", database_path,
    "--cross_validate_prior_focal_lengths", "1",
    "--min_calibrated_pair_ratio", "0.5", # 0.5
    "--reestimate_relative_pose", "1", # 1
    "--min_focal_length_ratio", "0.1",
    "--max_focal_length_ratio", "10",
    "--max_calibration_error", "2", # 2
    "--relpose_max_error", "1", # 1
    "--relpose_min_num_inliers", str(min_num_inliers), # 30
    "--relpose_min_inlier_ratio", "0.25", #str(min_inlier_ratio), # 0.25
]
run_subprocess(view_graph_calibrate_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
view_graph_calibrate_time = time.time() - view_graph_calibrate_start
logger.info(f"View graph calibrate done in {view_graph_calibrate_time:.2f} s")

# ========================= Visualize matches (optional) =========================
if args.visualize_matches:
    logger.info("Visualizing matches image pairs...")
    view_graph, cameras, images, feature_name = ReadColmapDatabase(database_path)
    if view_graph is None or cameras is None or images is None:
        logger.warning("Could not read database for visualization, skipping visualization step")
    else:
        vis_dir = os.path.join(output_path, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        print("vis_dir: ", vis_dir)
        visualize_image_pairs(view_graph, images, images_dir, vis_dir, num_pairs=100)
        print("Visualization completed.")

# ========================= Mapper / Bundle Adjustment =========================
# input("Press Enter to start mapper or refine...")
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper ...")
mapper_start = time.time()
if args.mapper == "acc":
    logger.info("Using incremental mapper with acc mode ...")
    mapper_cmd = get_incremental_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        ba_local_backend=args.ba_local_backend,
        ba_global_backend=args.ba_global_backend,
        ba_local_max_num_iterations=15,
        ba_local_max_refinements=2,
        ba_local_max_refinement_change=0.001,
        ba_global_max_num_iterations=25,
        ba_global_max_refinements=2,
        ba_global_frames_ratio=1.5,
        ba_global_points_ratio=1.5,
        ba_global_max_refinement_change=0.001,
        abs_pose_min_num_inliers=min_num_inliers,
        abs_pose_min_inlier_ratio=min_inlier_ratio,
        refine_focal_length=args.refine_focal_length,
        refine_principal_point=args.refine_principal_point,
        refine_extra_params=args.refine_extra_params
    )
elif args.mapper == "hierarchical":
    logger.info("Using hierarchical mapper...")
    mapper_cmd = get_hierarchical_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        abs_pose_min_num_inliers=min_num_inliers,
        abs_pose_min_inlier_ratio=min_inlier_ratio
    )
elif args.mapper == "hierarchical_acc":
    logger.info("Using hierarchical mapper with acc mode ...")
    mapper_cmd = get_hierarchical_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        ba_local_max_num_iterations=15,
        ba_local_max_refinements=2,
        ba_local_max_refinement_change=0.001,
        ba_global_max_num_iterations=25,
        ba_global_max_refinements=2,
        ba_global_frames_ratio=1.5,
        ba_global_points_ratio=1.5,
        ba_global_max_refinement_change=0.001,
        abs_pose_min_num_inliers=min_num_inliers,
        abs_pose_min_inlier_ratio=min_inlier_ratio,
        refine_focal_length=args.refine_focal_length,
        refine_principal_point=args.refine_principal_point,
        refine_extra_params=args.refine_extra_params
    )
elif args.mapper == "global":
    logger.info("Using global mapper...")
    mapper_cmd = get_global_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        ba_backend=args.ba_global_backend,
        min_num_inliers=min_num_inliers,
        ba_num_iterations=3,
        gp_max_num_iterations=args.gp_max_num_iterations,
        ba_ceres_max_num_iterations=args.ba_ceres_max_num_iterations,
        ra_max_rotation_error_deg=args.ra_max_rotation_error_deg,
        ra_max_rotation_error_final_deg=args.ra_max_rotation_error_final_deg,
        ra_refilt_outlier_paisrs_num=args.ra_refilt_outlier_pairs_num,
        max_normalized_reproj_error=args.max_normalized_reproj_error,
        globalMapper_min_tri_angle_deg=args.global_mapper_min_tri_angle_deg,
        tri_complete_max_reproj_error=15, #15
        tri_merge_max_reproj_error=15, #15
        tri_min_angle= 1, #1   
        track_min_num_views_per_track=args.track_min_num_views_per_track,
        refine_focal_length=args.refine_focal_length,
        refine_principal_point=args.refine_principal_point,
        refine_extra_params=args.refine_extra_params    
    )
elif args.mapper == "pose_prior_incremental":
    logger.info("Using prior-pose-based incremental mapper...")
    triangulate_cmd = get_points_triangulate_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        pose_prior=args.pose_prior,
        min_num_inliers=min_num_inliers,
        tri_create_max_angle_error = 2.0, # 2.0
        tri_continue_max_angle_error = 2.0, # 2.0
        tri_merge_max_reproj_error = 4.0, # 4.0
        tri_complete_max_reproj_error = 4.0, # 4.0
        tri_complete_max_transitivity = 5, # 5
        filter_max_reproj_error = 4.0, # 4.0
        filter_min_tri_angle = 1.5, # 1.5
        tri_re_max_angle_error = 5.0 # 5       
    )
    triangulate_start = time.time()
    run_subprocess(triangulate_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
    triangulate_time = time.time() - triangulate_start
    logger.info(f"Pose prior triangulation completed in {triangulate_time:.2f} s")

    # input("Press Enter to start bundle adjustment with pose prior...")
    ba_start = time.time()
    ba_cmd = get_ba_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        input_path=distorted_sparse_path,
        output_path=distorted_sparse_path,
        refine_focal_length=1,
        refine_principal_point=1,
        refine_extra_params=1,
        refine_rig_from_world=1,
        refine_sensor_from_rig=1,
        refine_points3D=1,
        min_track_length=args.track_min_num_views_per_track,
        max_num_iterations=200,
        max_linear_solver_iterations=200,
        gradient_tolerance=0.0001,
        use_gpu=use_gpu
    )    
    run_subprocess(ba_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)

    ba_time = time.time() - ba_start
    logger.info(f"Pose prior increment mapper bundle adjustment completed in {ba_time:.2f} s")
elif args.mapper == "pose_prior_global":
    logger.info("Using global mapper with pose prior...")
    max_normalized_reproj_error = args.max_normalized_reproj_error
    if prior_focal_length is not None:
        max_normalized_reproj_error = 3 / prior_focal_length
        logger.info(f"Setting max_normalized_reproj_error to {max_normalized_reproj_error:.4f} based on prior focal length {prior_focal_length:.2f}")
    mapper_cmd = get_pose_prior_global_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        prior_reconstruction_path=args.pose_prior,
        output_reconstruction_path=distorted_sparse_path,
        clear_points=1,
        use_gpu=use_gpu,
        min_num_inliers=min_num_inliers,
        ba_num_iterations=3,
        gp_max_num_iterations=args.gp_max_num_iterations,
        ba_ceres_max_num_iterations=args.ba_ceres_max_num_iterations,
        skip_rotation_averaging=0,
        skip_rotation_averaging_initialization=1,
        skip_track_establishment=0,
        skip_global_positioning=0,
        ba_skip_fixed_points_stage=1,
        ba_skip_fixed_rotation_stage=0,
        skip_bundle_adjustment=0, # 0
        skip_retriangulation=0, # 0        
        ba_skip_joint_optimization_stage=0,
        max_angular_reproj_error_deg=1.0,
        max_normalized_reproj_error=max_normalized_reproj_error,
        ra_max_rotation_error_deg=args.ra_max_rotation_error_deg,
        min_tri_angle_deg=args.global_mapper_min_tri_angle_deg,
        tri_complete_max_reproj_error=5, #15
        tri_merge_max_reproj_error=5, #15
        tri_min_angle= 2, #1       
        track_min_num_views_per_track=args.track_min_num_views_per_track
    )
elif args.mapper == "pos_prior":
    logger.info("Using incremental mapper with only images's position prior...")
    mapper_cmd = get_pos_prior_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        input_path="",
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        ba_local_backend=args.ba_local_backend,
        ba_global_backend=args.ba_global_backend,        
        ba_local_max_num_iterations = 25,
        ba_local_max_refinements = 2,
        ba_local_max_refinement_change = 0.001,
        ba_global_frames_ratio = 1.1,
        ba_global_points_ratio = 1.1,
        ba_global_max_num_iterations = 50,
        ba_global_max_refinements = 5,
        ba_global_max_refinement_change = 0.0005,
        ba_global_frames_freq = 500,
        ba_global_points_freq = 250000,
        min_num_inliers = 30,
        min_inlier_ratio = 0.25,
        overwrite_priors_covariance  = 1,
        prior_position_std_x  = 0.5,
        prior_position_std_y  = 0.5,
        prior_position_std_z  = 0.5      
    )
else:
    logger.info("Using incremental mapper...")
    mapper_cmd = get_incremental_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        ba_local_backend=args.ba_local_backend,
        ba_global_backend=args.ba_global_backend,        
        ba_local_max_num_iterations=25,
        ba_local_max_refinements=2,
        ba_local_max_refinement_change=0.001,
        ba_global_max_num_iterations=50, # 50
        ba_global_max_refinements=5,
        ba_global_frames_ratio=1.1,
        ba_global_points_ratio=1.1,
        ba_global_max_refinement_change=0.0005,
        abs_pose_min_num_inliers=min_num_inliers,
        abs_pose_min_inlier_ratio=min_inlier_ratio,
        filter_max_reproj_error=4.0, #4
        filter_min_tri_angle=1.5 #1.5
    )
if args.mapper != "pose_prior_incremental":
    run_subprocess(mapper_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
else:
    if args.refine_num > 0:
        refine_start = time.time()
        max_normalized_reproj_error = 0.01
        if prior_focal_length is not None:
            # max_normalized_reproj_error = 8 / prior_focal_length
            logger.info(f"Setting max_normalized_reproj_error to {max_normalized_reproj_error:.4f} based on prior focal length {prior_focal_length:.2f}")
        refine_cmd = get_reconstruction_refine_cmd(colmap_command=colmap_command,
            log_level=log_level,
            database_path=database_path,
            images_path=images_dir,
            prior_reconstruction_path=distorted_sparse_path,
            output_reconstruction_path=distorted_sparse_path,
            clear_points=0,
            use_gpu=use_gpu,
            min_num_inliers=min_num_inliers,
            ba_num_iterations=3,
            gp_max_num_iterations=100,
            ba_ceres_max_num_iterations=200,
            max_normalized_reproj_error=max_normalized_reproj_error,
            tri_complete_max_reproj_error= 5.0,
            tri_merge_max_reproj_error= 5.0)
        for it in range(0, args.refine_num):
            logger.info(f" {it + 1} / {args.refine_num} reconstruction refine")
            run_subprocess(refine_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
        refine_time = time.time() - refine_start        
mapper_time = time.time() - mapper_start + view_graph_calibrate_time
logger.info(f"Mapper done in {mapper_time:.2f} s")

## ======================== Final bundle adjustment with Ceres (optional) =========================
final_ceres_ba_time = 0
if args.ba_local_backend == "CASPAR" or args.ba_global_backend == "CASPAR":
    largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
    final_ceres_ba_start = time.time()
    ba_cmd = get_ba_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        input_path=largest_sparse_folder,
        output_path=largest_sparse_folder,
        refine_focal_length=args.refine_focal_length,
        refine_principal_point=args.refine_principal_point,
        refine_extra_params=args.refine_extra_params,
        refine_rig_from_world=1,
        refine_sensor_from_rig=1,
        refine_points3D=1,
        min_track_length=0,
        max_num_iterations=200,
        max_linear_solver_iterations=200,
        gradient_tolerance=0.0001,
        use_gpu=1,
        ba_backend="CERES"
    )    
    run_subprocess(ba_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)

    final_ceres_ba_time = time.time() - final_ceres_ba_start    
    mapper_time += final_ceres_ba_time
    logger.info(f"Final Ceres bundle adjustment done in {final_ceres_ba_time:.2f} s")

# ========================= Image undistortion =========================
# input("Press Enter to start image undistortion...")
start_undistort = time.time()
largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
logger.info(f"largest_sparse_folder: {largest_sparse_folder}")
images_pose = read_images_binary(os.path.join(largest_sparse_folder, "images.bin"))
registered_images_num = len(images_pose)
if args.undistort:
    input_subdirs = get_subfolders_names(images_dir)
    for subdir in input_subdirs:
        output_subdir_path = os.path.join(output_path, "images", subdir)
        os.makedirs(output_subdir_path, exist_ok=True)

    img_undist_cmd = [
        colmap_command, "image_undistorter",
        "--image_path", images_dir,
        "--input_path", largest_sparse_folder,
        "--output_path", output_path,
        "--output_type", "COLMAP"
    ]
    run_subprocess(img_undist_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)
undistort_time = time.time() - start_undistort
logger.info(f"Image undistortion done, used time: {undistort_time:.2f} s.")

# ========================= Organize sparse output to sparse/0 =============
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
# registered_images_num, _ = count_images_in_dir_recursive(os.path.join(output_path, "images"))
logger.info("Sparse output successfully organized into sparse/0.")

# ========================align sparse model to prior pose (if provided)=========================
get_aligned_sparse_model_cmd = get_model_align_cmd(colmap_command,
                    log_level= log_level,
                    database_path = database_path,
                    input_path = largest_sparse_folder,
                    output_path = largest_sparse_folder,
                    ref_is_gps = 0,
                    alignment_max_error = 10.0)

run_subprocess(get_aligned_sparse_model_cmd, logger, monitor_memory=args.monitor_memory, peak_memory=peak_memory)


# ========================= Rename iamge name in colmap results and move all images to output/images (if needed) =========================
if args.unify_output_images:
    logger.info("Moving images to output/images and updating names in sparse model ...")
    subdir_images_path = get_subfolders(os.path.join(output_path, "images"))
    for subdir in subdir_images_path:
        move_files(subdir, os.path.join(output_path, "images"))
        delete_directory(subdir)

    if len(subdir_images_path) > 0:
        logger.info(f"Moved images from {len(subdir_images_path)} subdirectories to output/images and deleted the subdirectories")
        images_result = os.path.join(output_path, "sparse/0", "images.bin")
        images_backup = os.path.join(output_path, "sparse/0", "images_backup.bin")
        os.rename(images_result, images_backup)
        images = read_images_binary(images_backup)
        for img_id, img in images.items():
            img_name = img.name
            img_path = os.path.join(output_path, "images", img_name)
            # Update image name in COLMAP results to match the input image name
            img_name = os.path.basename(img_path)
            new_path = os.path.join(output_path, "images", img_name)
            if not os.path.isfile(new_path):
                logger.warning(f"Image file {new_path} does not exist, skipping image name update for image ID {img_id}")
                continue
            # Use _replace() to create a new Image object since namedtuple fields are immutable
            images[img_id] = img._replace(name=img_name)

        # Write the updated model back to disk
        write_images_binary(images, images_result)


# --------------------------
# Timing summary
# --------------------------
total_time = time.time() - start_time
sparse_reconstruction_time = feature_extraction_time + feature_matching_time + mapper_time
pair_match_time = feature_matching_time / matched_image_pairs_num if matched_image_pairs_num > 0 else 0

logger.info("Done. Timing statistics:")
logger.info(f"  Feature extraction: {int(feature_extraction_time // 60)} min {feature_extraction_time % 60:.2f} s")
logger.info(f"  Feature matching: {int(feature_matching_time // 60)} min {feature_matching_time % 60:.2f} s")
logger.info(f"  Matched image pairs: {matched_image_pairs_num}")
logger.info(f"  Average time per image pair: {pair_match_time * 1000:.2f} ms")
logger.info(f"  View Grpha Calibrate time: {int(view_graph_calibrate_time // 60)} min {view_graph_calibrate_time % 60:.2f} s")
logger.info(f"  Triangulation time: {int(triangulate_time // 60)} min {triangulate_time % 60:.2f} s")
logger.info(f"  Bundle Adjustment time: {int(ba_time // 60)} min {ba_time % 60:.2f} s")
logger.info(f"  Final Ceres BA time: {int(final_ceres_ba_time // 60)} min {final_ceres_ba_time % 60:.2f} s")
logger.info(f"  Refine time: {int(refine_time // 60)} min {refine_time % 60:.2f} s")
logger.info(f"  Mapper: {int(mapper_time // 60)} min {mapper_time % 60:.2f} s")
logger.info(f"  Sparse reconstruction: {int(sparse_reconstruction_time // 60)} min {sparse_reconstruction_time % 60:.2f} s")
logger.info(f"  Image undistortion: {int(undistort_time // 60)} min {undistort_time % 60:.2f} s")
logger.info(f"  Total time: {int(total_time // 60)} min {total_time % 60:.2f} s")
logger.info(f"  Number of images registered: {registered_images_num} / {input_img_num}")

# ---- cumulative peak memory summary ----
if peak_memory and (peak_memory.get("cpu_mb", 0) > 0 or peak_memory.get("gpu_mb")):
    parts = []
    cpu_mb = peak_memory.get("cpu_mb", 0)
    if cpu_mb > 0:
        parts.append(f"Peak CPU RAM: {cpu_mb:.1f} MB")
    gpu_mb = peak_memory.get("gpu_mb", {})
    if gpu_mb:
        gpu_strs = [f"GPU-{idx}: {used:.1f} MB" for idx, used in sorted(gpu_mb.items())]
        parts.append(f"Peak GPU VRAM: {', '.join(gpu_strs)}")
    if parts:
        logger.info(f"  ** Cumulative peak memory — {' | '.join(parts)} **")
