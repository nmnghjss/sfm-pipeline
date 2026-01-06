#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python implementation of the bash script written by Romuald Perrot
# Created by @vins31
# Modified by Pierre Moulon
#
# this script is for easy use of OpenMVG
#
# usage : python openmvg.py image_dir output_dir
#
# image_dir is the input directory where images are located
# output_dir is where the project must be saved
#
# if output_dir is not present script will create it
#

# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/home/nmnghjss/Apps/OpenMVG/bin"

# Indicate the openMVG camera sensor width directory
# CAMERA_SENSOR_WIDTH_DIRECTORY = (
#     "/home/nmnghjss/Codes/openMVG/src/software/SfM"
#     + "/../../openMVG/exif/sensor_width_database"
# )

import os
import shutil
import subprocess
import sys
import time
import PIL.Image as Image

if len(sys.argv) < 3:
    print("Usage %s image_dir output_dir" % sys.argv[0])
    sys.exit(1)


start_time = time.time()

input_dir = sys.argv[1]
output_dir = sys.argv[2]
focal_length = sys.argv[3] if len(sys.argv) > 3 else None

matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_global")

print("Using input dir  : ", input_dir)
print("      output_dir : ", output_dir)

# 0. get focal length if not provided
if focal_length is None:
    print("0. Estimate focal length from image EXIF data")
    # 遍历文件夹中的所有文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_extensions:
                img =  Image.open(file_path)
                width, height = img.size
                focal_length = str(int(1.2 * max(width, height)))
                break
    if focal_length is None:
        focal_length = "1500"  # default value
    print(f"   Estimated focal length: {focal_length}")


# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

print("1. Intrinsics analysis")
pIntrisics = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),
        "-i",
        input_dir,
        "-o",
        matches_dir,
        "-f",
        focal_length,
    ]
)
pIntrisics.wait()

print("2. Compute features")
pFeatures = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),
        "-i",
        matches_dir + "/sfm_data.json",
        "-o",
        matches_dir,
        "-m",
        "SIFT",
    ]
)
pFeatures.wait()

print("3. Compute matching pairs")
pPairs = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"),
        "-i",
        matches_dir + "/sfm_data.json",
        "-o",
        matches_dir + "/pairs.bin",
    ]
)
pPairs.wait()

print("4. Compute matches")
pMatches = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),
        "-i",
        matches_dir + "/sfm_data.json",
        "-p",
        matches_dir + "/pairs.bin",
        "-o",
        matches_dir + "/matches.putative.bin",
    ]
)
pMatches.wait()

print("5. Filter matches")
pFiltering = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"),
        "-i",
        matches_dir + "/sfm_data.json",
        "-m",
        matches_dir + "/matches.putative.bin",
        "-g",
        "e",
        "-o",
        matches_dir + "/matches.e.bin",
    ]
)
pFiltering.wait()

# Create the reconstruction if not present
if not os.path.exists(reconstruction_dir):
    os.mkdir(reconstruction_dir)

print("6. Do Global reconstruction")
pRecons = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"),
        "--sfm_engine",
        "GLOBAL",
        "--input_file",
        matches_dir + "/sfm_data.json",
        "--match_file",
        matches_dir + "/matches.e.bin",
        "--output_dir",
        reconstruction_dir,
    ]
)
pRecons.wait()

print("7. Colorize Structure")
pRecons = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),
        "-i",
        reconstruction_dir + "/sfm_data.bin",
        "-o",
        os.path.join(reconstruction_dir, "colorized.ply"),
    ]
)
pRecons.wait()

print("8. Convert to Colmap format")
colpath_distorted = os.path.join(output_dir, "colmap-distorted")
os.makedirs(colpath_distorted, exist_ok=True)
pColmap = subprocess.Popen(
    [
        os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2Colmap"),
        "-i",
        reconstruction_dir + "/sfm_data.bin",
        "-o",
        colpath_distorted,
    ]
)
pColmap.wait()

print("9. Undistort images")
images_path = input_dir
input_path = os.path.join(output_dir, "colmap-distorted")
colpath_undistorted = os.path.join(output_dir, "colmap-undistorted")
os.makedirs(colpath_undistorted, exist_ok=True)
img_undist_cmd = [
    "colmap",
    "image_undistorter",
    "--image_path",
    images_path,
    "--input_path",
    colpath_distorted,
    "--output_path",
    colpath_undistorted,
    "--output_type",
    "COLMAP",
]
pColmap = subprocess.Popen(img_undist_cmd)
pColmap.wait()


sparse_output_path = os.path.join(colpath_undistorted, "sparse")
os.makedirs(os.path.join(sparse_output_path, "0"), exist_ok=True)
for item in os.listdir(sparse_output_path):
    src_path = os.path.join(sparse_output_path, item)
    dst_path = os.path.join(sparse_output_path, "0", item)
    if os.path.isdir(src_path):
        if item == "0":
            continue
        for f in os.listdir(src_path):
            f_src = os.path.join(src_path, f)
            f_dst = os.path.join(colpath_undistorted, "0", f)
            shutil.move(f_src, f_dst)
        os.rmdir(src_path)
    elif os.path.isfile(src_path):
        shutil.move(src_path, dst_path)


end_time = time.time()
used_time = end_time - start_time

print(f"all openmvg sfm time: {used_time:.1f} seconds")
