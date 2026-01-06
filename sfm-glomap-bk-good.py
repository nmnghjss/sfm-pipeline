
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
parser.add_argument("--output_path", "-o", default="", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--single_camera", "-sc",default="0", type=str)
parser.add_argument("--single_fold", "-sf", default="1", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--acc", action="store_true", help="Use accelerated parameters for COLMAP")
parser.add_argument("--images_folds", "-I", nargs='+', default=None,
                    help="List of image folders (relative to source_path or absolute). Optional.")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
args = parser.parse_args()


# --------------------------
# Helper functions
# --------------------------
def resource_path(relative_path: str) -> str:
    """Get absolute path for packaged or development scripts."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def run_subprocess(cmd: list, log_path: str):
    """
    Run a subprocess command, print stdout/stderr in real-time and save to log file.
    Windows compatible.
    """
    logger.info(f"Running command: {' '.join(cmd)}")
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            shell=False  # Windows, using list command
        )

        # Real-time printing and logging
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(line)
                log_file.write(line + "\n")
        process.wait()

    if process.returncode != 0:
        logger.error(f"Command failed with code {process.returncode}. See {log_path} for details.")
        raise subprocess.CalledProcessError(process.returncode, cmd)


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

def count_image_files(folder_path: str) -> int:
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


# --------------------------
# Paths and executables
# --------------------------
current_path = resource_path(".")
colmap_path = "colmap"
colmap_command = args.colmap_executable if args.colmap_executable else colmap_path

use_gpu = 0 if args.no_gpu else 1

output_path = args.output_path if args.output_path else args.source_path
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
log_dir = args.output_path if args.output_path else args.source_path
os.makedirs(log_dir, exist_ok=True)  # <--- 确保目录存在
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
mapper_time = 0

# --------------------------
# Feature extraction & matching
# --------------------------
distorted_sparse_path = os.path.join(output_path, "distorted/sparse")
os.makedirs(distorted_sparse_path, exist_ok=True)

database_path = os.path.join(output_path, "distorted/database.db")
images_path = os.path.join(args.source_path, "input")

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
run_subprocess(feat_extraction_cmd, feat_extraction_log)
feature_extraction_time = time.time() - t0
logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

# --- Feature matching ---
match_log_path = os.path.join(output_path, "matcher.log")
logger.info("Starting feature matching...")
t1 = time.time()
if args.acc:
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
else:
    feat_matching_cmd = [
        colmap_command, "exhaustive_matcher",
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--log_level", str(log_level),
    ]
run_subprocess(feat_matching_cmd, match_log_path)
feature_matching_time = time.time() - t1
logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

# --------------------------
# Mapper / Bundle Adjustment
# --------------------------
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper...")
t2 = time.time()
if args.acc:
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
else:
    mapper_cmd = [
        colmap_command, "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--log_level", str(log_level),
    ]
run_subprocess(mapper_cmd, mapper_log_path)
mapper_time = time.time() - t2
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
run_subprocess(img_undist_cmd, undistorted_log_path)
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

img_num = count_image_files(os.path.join(output_path, "images"))

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
logger.info(f"  Number of images registered: {img_num} ")