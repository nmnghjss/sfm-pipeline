
import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
import subprocess
from typing import Optional

# --------------------------
# Argument parser
# --------------------------
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--only", action='store_true', help="Only process the source_path, not subfolders")
parser.add_argument("--output_path", "-o", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--single_camera", "-sc",default="0", type=str)
parser.add_argument("--single_fold", "-sf", default="1", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--alg", default="default", type=str, help="algorithm to use: default, acc, glomap")
parser.add_argument("--images_folds", "-I", nargs='+', default=None,
                    help="List of image folders (relative to source_path or absolute). Optional.")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
args = parser.parse_args()


# =========================================Helper functions===============================

def resource_path() -> str:
    """Get absolute path for packaged or development scripts."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

def run_subprocess(cmd: list, logger: Optional[logging.Logger] = None) -> int:
    """
    Run a subprocess command, print stdout/stderr in real-time and save to log file.
    Windows compatible. Uses the provided logger (per-folder) if given.
    
    Returns:
        0 if the command succeeded, -1 if it failed or crashed.
    """
    if logger is None:
        logger = logging.getLogger("sfm")

    logger.info(f"Running command: {' '.join(cmd)}")

    # Check if the logger has a StreamHandler attached that writes to stdout (robust check)
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, getattr(sys, "__stdout__", None))
        for h in logger.handlers
    )

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            shell=False  # Windows, using list command
        )
    except Exception as e:
        logger.error(f"Error: Failed to start process: {e}")
        return -1

    # Real-time printing and logging
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # If logger writes to stdout, rely on it (formatted). Otherwise, print to terminal (flush) to guarantee visibility.
                if has_stdout_handler:
                    logger.info(line)
                else:
                    print(line, flush=True)
                    logger.info(line)
    except Exception as e:
        logger.error(f"Error reading process output: {e}")
        try:
            process.kill()
        except:
            logger.error("Error: Failed to kill process after output read error.")
            pass
        return -1

    process.wait()

    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}.")
        return -1
    else:
        logger.info("Command completed successfully.")
        return 0


def configure_logger(output_path: str, level: int, logger_name: Optional[str] = None) -> logging.Logger:
    """Create or reconfigure a named logger for each run to avoid duplicate handlers.

    If logger_name is provided, a distinct logger will be used for that data folder, enabling
    separate log files and isolation between iterations.
    """
    if logger_name is None:
        logger_name = "sfm"

    # Normalize level: if user passed 0 or invalid small number, default to INFO
    if level <= 0:
        level = logging.INFO
    else:
        level = int(level)

    logger = logging.getLogger(logger_name)
    # Remove existing handlers to avoid duplicate logs when reconfiguring
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)
    logger.propagate = False

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler -> explicitly send to stdout so it appears in terminal
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

    # File handler (overwrite per run)
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "run-sfm.log")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)

    # Emit a small verification message (both print and logger) so terminal visibility can be tested
    print(f"[sfm] Logger '{logger_name}' configured for '{output_path}' (level={level})", flush=True)
    logger.info(f"Logger '{logger_name}' configured (level={level}) for '{output_path}'")

    return logger


def get_subfolders(parent_dir: str) -> list:
    """Return a list of subfolder paths in the given parent directory."""
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Path not found or not a directory: {parent_dir}")
    subfolders = []
    for name in os.listdir(parent_dir):
        sub_path = os.path.join(parent_dir, name)
        if os.path.isdir(sub_path):
            subfolders.append(sub_path)
    return subfolders


def get_largest_subfolder(parent_dir: str) -> Optional[str]:
    """Return the subfolder with the largest total file size."""
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Path not found or not a directory: {parent_dir}")
    max_size = -1
    largest_subfolder = None
    for name in os.listdir(parent_dir):
        sub_path = os.path.join(parent_dir, name)
        if not os.path.isdir(sub_path):
            continue
        total_size = 0
        for root, _, files in os.walk(sub_path):
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except OSError:
                    pass
        if total_size > max_size:
            max_size = total_size
            largest_subfolder = sub_path
    return largest_subfolder

def count_images_in_dir(folder_path: str) -> int:
    """
    统计指定文件夹路径下的图像文件数量（仅当前文件夹，不包含子文件夹）
    """
    # 定义常见的图像文件扩展名（转小写，方便统一判断）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

    image_count = 0

    if not os.path.exists(folder_path):
        print(f"错误：文件夹路径 '{folder_path}' 不存在！")
        return 0
    if not os.path.isdir(folder_path):
        print(f"错误：'{folder_path}' 不是一个有效的文件夹路径！")
        return 0

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_extensions:
                image_count += 1

    return image_count


def count_images_in_dir_recursive(root_dir, image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

    from pathlib import Path
    root = Path(root_dir)
    # 转为小写便于比较
    image_extensions = {ext.lower() for ext in image_extensions}
    
    count = 0
    for file in root.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            count += 1
    return count

# =========================================Paths and executables=========================================

current_path = resource_path()
colmap_path = "colmap"
glomap_path = "glomap"
colmap_command = args.colmap_executable if args.colmap_executable else colmap_path
glomap_command = args.glomap_executable if args.glomap_executable else glomap_path

use_gpu = 0 if args.no_gpu else 1

if args.only:
    datas_folder = [args.source_path]
else:
    datas_folder = get_subfolders(args.source_path)
    datas_folder = sorted(datas_folder)

output_dir_name = os.path.basename(os.path.normpath(args.output_path))

# =========================================Executables============================================
for data_folder in datas_folder:

    output_path = os.path.join(data_folder, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # --------------------------
    # Logging setup (per-folder logger)
    # --------------------------
    log_level = args.log_level
    # Use a unique logger name per data folder to ensure logs are isolated
    folder_name = os.path.basename(os.path.normpath(data_folder))
    logger_name = f"sfm.{folder_name}"
    logger = configure_logger(output_path, log_level, logger_name)


    # --------------------------
    # Timing variables
    # --------------------------
    start_time = time.time()
    feature_extraction_time = 0
    feature_matching_time = 0
    mapper_time = 0

    # --------------------------
    # Feature extraction & matching
    # --------------------------
    distorted_sparse_path = os.path.join(output_path, "distorted/sparse")
    os.makedirs(distorted_sparse_path, exist_ok=True)

    database_path = os.path.join(output_path, "distorted/database.db")
    images_path = os.path.join(data_folder, "input")
    input_images_num = count_images_in_dir_recursive(images_path)

    # --- Feature extraction ---
    feat_extraction_log = os.path.join(output_path, "feature_extraction.log")
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_path,
        "--ImageReader.single_camera_per_image", str(args.single_image),
        "--ImageReader.single_camera_per_fold", str(args.single_fold),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.camera,
        # "--SiftExtraction.peak_threshold", "0.00667", # 0.00667
        # "--SiftExtraction.max_num_orientations", "2", # 2
        "--FeatureExtraction.use_gpu", str(use_gpu),
        "--log_level", str(log_level),
    ]
    logger.info("Starting feature extraction...")
    t0 = time.time()
    ret = run_subprocess(feat_extraction_cmd, logger)
    feature_extraction_time = time.time() - t0
    if ret != 0:
        logger.error("Feature extraction failed. Skipping to next data folder.")
        continue
    logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

    # --- Feature matching ---
    match_log_path = os.path.join(output_path, "matcher.log")
    logger.info("Starting feature matching...")
    t1 = time.time()
    if args.alg == "glomap":
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--ExhaustiveMatching.block_size", "100", # 50
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.guided_matching", "0", # 0
            "--SiftMatching.max_ratio", "0.8", # 0.8
            "--SiftMatching.max_distance", "0.55", # 0.7
            "--TwoViewGeometry.min_num_inliers", "30", # 15
            "--TwoViewGeometry.max_error", "4", # 4
            "--TwoViewGeometry.confidence", "0.9999", # 0.999
            "--TwoViewGeometry.min_inlier_ratio", "0.9", # 0.25
            "--TwoViewGeometry.detect_watermark", "0", # 1
            "--TwoViewGeometry.filter_stationary_matches", "1", # 0
            "--TwoViewGeometry.compute_relative_pose", "0", # 0
            "--log_level", str(log_level), # 0
        ]
    elif args.alg == "acc":
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--ExhaustiveMatching.block_size", "100", # 50
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--FeatureMatching.guided_matching", "0", # 0
            "--SiftMatching.max_ratio", "0.7", # 0.8
            "--SiftMatching.max_distance", "0.6", # 0.7
            "--TwoViewGeometry.min_num_inliers", "100", # 15
            "--TwoViewGeometry.max_error", "4", # 4
            "--TwoViewGeometry.confidence", "0.9999", # 0.999
            "--TwoViewGeometry.min_inlier_ratio", "0.7", # 0.25
            "--TwoViewGeometry.detect_watermark", "0", # 1
            "--TwoViewGeometry.filter_stationary_matches", "1", # 0
            "--TwoViewGeometry.compute_relative_pose", "0", # 0
            "--log_level", str(log_level), # 0
        ]    
    else:
        feat_matching_cmd = [
            colmap_command, "exhaustive_matcher",
            "--database_path", database_path,
            "--FeatureMatching.use_gpu", str(use_gpu),
            "--log_level", str(log_level),
        ]
    ret = run_subprocess(feat_matching_cmd, logger)
    feature_matching_time = time.time() - t1
    if ret != 0:
        logger.error("Feature matching failed. Skipping to next data folder.")
        continue
    logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

    # -------------------------- Mapper / Bundle Adjustment --------------------------
    logger.info("==========================================================================")
    logger.info("Starting Mapper / Bundle Adjustment ...")
    logger.info("==========================================================================")
    t2 = time.time()
    if args.alg == "glomap":
        mapper_cmd = [
            "glomap", "mapper",
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
            "--GlobalPositioning.thres_loss_function", "0.2", # 0.1
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
            "--Triangulation.min_angle", "3", # 1
            "--Triangulation.min_num_matches", "30", # 15
            "--Thresholds.max_angle_error", "1",
            "--Thresholds.max_reprojection_error", "0.01",
            "--Thresholds.min_triangulation_angle", "3", # 1
            "--Thresholds.max_epipolar_error_E", "1",
            "--Thresholds.max_epipolar_error_F", "4",
            "--Thresholds.max_epipolar_error_H", "4",
            "--Thresholds.min_inlier_num", "30", # 30
            "--Thresholds.min_inlier_ratio", "0.9", # 0.25
            "--Thresholds.max_rotation_error", "10"
        ]
    elif args.alg == "acc":
        mapper_cmd = [
            colmap_command, "mapper",
            "--database_path", database_path,
            "--image_path", images_path,
            "--output_path", distorted_sparse_path,
            "--Mapper.num_threads", "-1",
            "--Mapper.min_num_matches", "100", # 15
            "--Mapper.init_num_trials", "200", # 200
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
            "--Mapper.ba_global_max_num_iterations", "20", # 50
            "--Mapper.ba_global_max_refinements", "2", # 5
            "--Mapper.ba_global_max_refinement_change", "0.001", # 0.0005
            "--Mapper.ba_refine_focal_length", "1", # 1
            "--Mapper.ba_refine_principal_point", "0", # 0
            "--Mapper.ba_refine_extra_params", "1", # 1
            "--Mapper.max_extra_param", "0.3", # 1
            "--Mapper.tri_min_angle", "2.0", # 1.5
            "--Mapper.tri_create_max_angle_error", "2", # 2
            "--Mapper.tri_merge_max_reproj_error", "4", # 4
            "--Mapper.filter_max_reproj_error", "4", # 4
            "--Mapper.max_reg_trials", "3", # 3
            "--log_level", str(log_level),
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
    ret = run_subprocess(mapper_cmd, logger)
    mapper_time = time.time() - t2
    if ret != 0:
        logger.error("Mapper failed. Skipping to next data folder.")
        continue
    logger.info(f"Mapper done in {mapper_time:.2f} s")

    # --------------------------
    # Image undistortion
    # --------------------------
    largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
    undistorted_log_path = os.path.join(output_path, "image_undistorter.log")
    img_undist_cmd = [
        colmap_command, "image_undistorter",
        "--image_path", images_path,
        "--input_path", largest_sparse_folder,
        "--output_path", output_path,
        "--output_type", "COLMAP"
    ]
    ret = run_subprocess(img_undist_cmd, logger)
    if ret != 0:
        logger.error("Image undistortion failed. Skipping to next data folder.")
        continue
    logger.info("Image undistortion done.")

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

    logger.info("Done. Timing statistics:")
    logger.info(f"  Feature extraction: {int(feature_extraction_time // 60)} min {feature_extraction_time % 60:.2f} s")
    logger.info(f"  Feature matching: {int(feature_matching_time // 60)} min {feature_matching_time % 60:.2f} s")
    logger.info(f"  Mapper: {int(mapper_time // 60)} min {mapper_time % 60:.2f} s")
    logger.info(f"  Sparse reconstruction: {int(sparse_reconstruction_time // 60)} min {sparse_reconstruction_time % 60:.2f} s")
    logger.info(f"  Total time: {int(total_time // 60)} min {total_time % 60:.2f} s")
    logger.info(f"  Number of images registered: {img_num}/{input_images_num}")