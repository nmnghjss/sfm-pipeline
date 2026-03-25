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

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"

use_gpu = 1 if not args.no_gpu else 0

os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

## Feature extraction
feat_extracton_cmd = colmap_command + " feature_extractor "\
    "--database_path " + args.source_path + "/distorted/database.db \
    --image_path " + args.source_path + "/input \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model " + args.camera 
exit_code = os.system(feat_extracton_cmd)
if exit_code != 0:
    logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
    exit(exit_code)

## Feature matching
feat_matching_cmd = colmap_command + " exhaustive_matcher \
    --database_path " + args.source_path + "/distorted/database.db"
exit_code = os.system(feat_matching_cmd)
if exit_code != 0:
    logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
    exit(exit_code)

### Bundle adjustment
# The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
mapper_cmd = (colmap_command + " mapper \
    --database_path " + args.source_path + "/distorted/database.db \
    --image_path "  + args.source_path + "/input \
    --output_path "  + args.source_path + "/distorted/sparse \
    --Mapper.ba_global_function_tolerance=0.000001")
exit_code = os.system(mapper_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)


# --------------------------
# Image undistortion
# --------------------------

def get_largest_subfolder(parent_dir: str):
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

distorted_sparse_path = os.path.join(args.source_path, "distorted/sparse")
images_path = os.path.join(args.source_path, "input")
largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
print(f"largest subfolder {largest_sparse_folder}")
output_path = args.source_path
img_undist_cmd = [
    colmap_command, "image_undistorter",
    "--image_path", images_path,
    "--input_path", largest_sparse_folder,
    "--output_path", output_path,
    "--output_type", "COLMAP"
]

img_undist_cmd = colmap_command +  " image_undistorter " + " --image_path " + images_path \
    + " --input_path " + largest_sparse_folder + " --output_path " + output_path + " " \
    + " --output_type " + " COLMAP "

os.system(img_undist_cmd)
    
print("Image undistortion done.")

# --------------------------
# Organize sparse output to sparse/0
# --------------------------
sparse_output_path = os.path.join(output_path, "sparse")
os.makedirs(os.path.join(sparse_output_path, "0"), exist_ok=True)

print("Organizing sparse output files into sparse/0 ...")
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

print("Done.")
