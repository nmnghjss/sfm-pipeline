
import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from pathlib import Path
from utils import *
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
parser.add_argument("--default", action="store_true", help="Use default parameters for COLMAP")
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


# --------------------------
# Paths and executables
# --------------------------
current_path = resource_path(".")
os_type = check_operating_system()
print(f"Detected operating system: {os_type}")
if os_type == 'Windows':
    colmap_path = os.path.join(current_path, "colmap-x64-windows-cuda-3.13.0/bin/colmap.exe")
else:
    colmap_path = "colmap"
colmap_command = args.colmap_executable if args.colmap_executable else colmap_path

use_gpu = 0 if args.no_gpu else 1

if args.output_path == "":
    output_path = args.source_path
else:
    output_path = args.output_path
    if not Path(output_path).is_absolute():
        output_path = os.path.join(args.source_path, output_path)

print("Output path:", output_path)
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
run_subprocess(feat_extraction_cmd, logger)
feature_extraction_time = time.time() - t0
logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

# --- Feature matching ---
match_log_path = os.path.join(output_path, "matcher.log")
logger.info("Starting feature matching...")
t1 = time.time()
if not args.default:
    feat_matching_cmd = [
        colmap_command, "exhaustive_matcher",
        "--database_path", database_path,
        "--ExhaustiveMatching.block_size", "100", # 50
        "--FeatureMatching.use_gpu", str(use_gpu),
        "--FeatureMatching.guided_matching", "0", # 0
        "--SiftMatching.max_ratio", "0.8", # 0.8
        "--SiftMatching.max_distance", "0.7", # 0.7
        "--TwoViewGeometry.min_num_inliers", "25", # 15
        "--TwoViewGeometry.max_error", "4", # 4
        "--TwoViewGeometry.confidence", "0.9999", # 0.999
        "--TwoViewGeometry.min_inlier_ratio", "0.5", # 0.25
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
run_subprocess(feat_matching_cmd, logger)
feature_matching_time = time.time() - t1
logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

# --------------------------
# Mapper / Bundle Adjustment
# --------------------------
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper...")
t2 = time.time()
if not args.default:
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
run_subprocess(mapper_cmd, logger)
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
run_subprocess(img_undist_cmd, logger)
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
logger.info(f"  Number of images registered: {img_num} ")