#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
import time

# Record start time
start_time = time.time()

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--output_path", "-o", default="", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "D:\Programs\colmap-x64-windows-cuda-3.8\COLMAP.bat"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

output_dir_name = os.path.basename(os.path.normpath(args.output_path))
output_dir = os.path.join(args.source_path, output_dir_name)
os.makedirs(output_dir, exist_ok=True)
distorted_sparse_path = os.path.join(output_dir, "distorted/sparse")
os.makedirs(distorted_sparse_path, exist_ok=True)

database_path = os.path.join(output_dir, "distorted/database.db")
images_path = os.path.join(args.source_path, "input")


if not args.skip_matching:

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + database_path + " \
        --image_path " + images_path + " \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera 
        # --FeatureExtraction.use_gpu " + str(use_gpu)
    feat_extraction_start = time.time()
    exit_code = os.system(feat_extracton_cmd)
    feature_extraction_time = time.time() - feat_extraction_start
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + output_dir + "/distorted/database.db" 
        # --FeatureMatching.use_gpu " + str(use_gpu)
    feat_matching_start = time.time()
    exit_code = os.system(feat_matching_cmd)
    feature_matching_time = time.time() - feat_matching_start
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + database_path + " \
        --image_path "  + images_path + " \
        --output_path "  + output_dir + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    mapper_start = time.time()
    # exit_code = os.system(mapper_cmd)
    mapper_time = time.time() - mapper_start
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + images_path + " \
    --input_path " + output_dir + "/distorted/sparse/0 \
    --output_path " + output_dir + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(output_dir + "/sparse")
os.makedirs(output_dir + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(output_dir, "sparse", file)
    destination_file = os.path.join(output_dir, "sparse", "0", file)
    shutil.move(source_file, destination_file)

end_time = time.time()
elapsed_time = end_time - start_time
sparse_reconstruction_time = feature_extraction_time + feature_matching_time + mapper_time
# Print timing statistics
print(f"Done. Timing statistics:")
print(f"  Feature extraction: {int(feature_extraction_time // 60)} min {feature_extraction_time % 60:.2f} s")
print(f"  Feature matching: {int(feature_matching_time // 60)} min {feature_matching_time % 60:.2f} s")
print(f"  Mapper: {int(mapper_time // 60)} min {mapper_time % 60:.2f} s")
print(f"  Sparse reconstruction: {int(sparse_reconstruction_time // 60)} min {sparse_reconstruction_time % 60:.2f} s")
print(f"  Total time: {int(elapsed_time // 60)} min {elapsed_time % 60:.2f} s")