from SuperPointDetectors import get_super_points_from_scenes_return
from matchers import mutual_nn_matcher, mutual_nn_matcher_torch
import cv2
import os, time
import numpy as np
import argparse
from database import COLMAPDatabase
from utils import *
import torch
from mapper import mapper as colmap_mapper
import shutil

camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10}

def get_init_cameraparams(width, height, modelId):
    f = max(width, height) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([f, f, cx, cy])
    elif modelId == 2 or modelId == 6:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 8:
        return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0])
    elif modelId == 9:
        return np.array([f, f, cx, cy, 0.0])
    return np.array([f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def init_cameras_database(db, images_path, cameratype, single_camera):
    print("init cameras database ......................................")
    images_name = []
    width = None
    height = None
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    for name in sorted(os.listdir(images_path)):
            if is_image(name):
                images_name.append(name)
                if width is None:
                    img = cv2.imread(os.path.join(images_path, name))
                    height, width = img.shape[:2]
    cameraModel = camModelDict[cameratype]
    params = get_init_cameraparams(width, height, cameraModel)
    if single_camera:
        db.add_camera(cameraModel, width, height, params, camera_id=0)
    for i, name in enumerate(images_name):
        if single_camera:
            db.add_image(name, 0, image_id=i)
            continue
        db.add_camera(cameraModel, width, height, params, camera_id=i)
        db.add_image(name, i, image_id=i)
    return images_name

def import_feature(db, images_path, images_name):
    print("feature extraction by super points ...........................")
    sps = get_super_points_from_scenes_return(images_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)

    return sps

def import_feature_from_sps(db, sps, images_name):
    print("feature extraction by super points ...........................")
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)


def match_features(db, sps, images_name, match_list_path):
    print("match features by sequential match............................")
    # sequential match
    step_range = [1, 2, 3, 5, 8, 13, 21, 44, 65, 109, 174, 210]
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')
    for step in step_range:
        for i in range(0, num_images - step):
            match_list.write("%s %s\n" % (images_name[i], images_name[i + step]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[i + step]]['descriptors'] * 1.0
            matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
            db.add_matches(i, i + step, matches)
    match_list.close()

def match_features_exhaustive(db, sps, images_name, match_list_path):
    print("match features by exhaustive match............................")
    # exhaustive match
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')
    for i in range(0, num_images):
        for j in range(i + 1, num_images):
            match_list.write("%s %s\n" % (images_name[i], images_name[j]))
            D1 = sps[images_name[i]]['descriptors'] * 1.0
            D2 = sps[images_name[j]]['descriptors'] * 1.0
            matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
            db.add_matches(i, j, matches)
    match_list.close()

def match_features_exhaustive_v2(db, sps, images_name, match_list_path):
    print("match features by exhaustive match............................")

    # Convert all descriptors to torch tensors on GPU.
    for name in images_name:
        sps[name]['descriptors'] = torch.tensor(sps[name]['descriptors'], device="cuda", dtype=torch.float32)

    # Exhaustive match
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')
    for i in range(0, num_images):
        for j in range(i + 1, num_images):
            match_list.write("%s %s\n" % (images_name[i], images_name[j]))
            D1 = sps[images_name[i]]['descriptors']
            D2 = sps[images_name[j]]['descriptors']
            matches = mutual_nn_matcher_torch(D1, D2).astype(np.uint32)
            db.add_matches(i, j, matches)
    match_list.close()


def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    duration = end - start
    print("[%s] cost %f s" % (cmd, duration))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mapper(projpath, images_path):
    database_path = os.path.join(projpath, "database.db")
    colmap_sparse_path = os.path.join(projpath, "sparse")
    makedir(colmap_sparse_path)

    mapper = "colmap mapper --database_path %s --image_path %s --output_path %s" % (
        database_path, images_path, colmap_sparse_path
    )
    operate(mapper)

def geometric_verification(database_path, match_list_path):
    print("Running geometric verification..................................")
    cmd = "colmap matches_importer --database_path %s --match_list_path %s --match_type pairs" % (
        database_path, match_list_path
    )
    operate(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points colmap')
    parser.add_argument("--projpath", required=True, type=str)
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="")
    parser.add_argument("--log_level", type=int, required=False, default=0)    
    parser.add_argument("--cameraModel", type=str, required=False, default="OPENCV")
    parser.add_argument("--images_path", required=False, type=str, default="input")
    parser.add_argument("--single_camera", action='store_true')
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--glomap_executable", default="", type=str) 
    parser.add_argument("--alg", type=str, required=False, default="acc")   

    args = parser.parse_args()

    # colmap_path = "colmap"
    colmap_path = "D:\\Programs\\colmap-x64-windows-cuda-3.13.0\\bin\\colmap.exe"
    colmap_command = args.colmap_executable if args.colmap_executable else colmap_path
    # colmap_path = "colmap"
    colmap_command = args.colmap_executable if args.colmap_executable else colmap_path
    glomap_path = "glomap"
    glomap_command = args.glomap_executable if args.glomap_executable else glomap_path


    if args.output_dir == "":
        output_dir = args.projpath
    else:
        output_dir = os.path.join(args.projpath, os.path.basename(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    clear_folder(output_dir)

    start_time = time.time()
    distorted_sparse_path = os.path.join(output_dir, "distorted/sparse")
    os.makedirs(distorted_sparse_path, exist_ok=True)    

    log_level = args.log_level
    folder_name = os.path.basename(os.path.normpath(output_dir))
    logger_name = f"sfm.{folder_name}"
    logger = configure_logger(output_dir, log_level, logger_name)


    database_path = os.path.join(output_dir, "database.db")
    match_list_path = os.path.join(distorted_sparse_path, "image_pairs_to_match.txt")

    images_path = os.path.join(args.projpath, args.images_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    input_images_num = count_images_in_dir_recursive(images_path)

    feature_start_time = time.time()
    images_name = init_cameras_database(db, images_path, args.cameraModel, args.single_camera)
    sps = import_feature(db, images_path, images_name)
    feature_extraction_time = time.time() - feature_start_time
    logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")
    a = input("Press Enter to start feature matching...")

    feature_matching_start_time = time.time()
    # match_features(db, sps, images_name, match_list_path)
    match_features_exhaustive_v2(db, sps, images_name, match_list_path)
    feature_matching_time = time.time() - feature_matching_start_time
    logger.info(f"Feature matching done in {feature_matching_time:.2f} s")
    db.commit()
    db.close()


    logger = configure_logger(output_dir, log_level, logger_name)

    geometric_verification(database_path, match_list_path)

    mapper_time = 0
    mapper_start_time = time.time()
    # mapper(args.projpath, images_path)
    colmap_mapper(
        database_path=database_path,
        images_path=images_path,
        distorted_sparse_path=distorted_sparse_path,
        logger=logger,
        log_level=log_level,
        colmap_command=colmap_command,
        alg = args.alg
    )
    mapper_time = time.time() - mapper_start_time
    logger.info(f"Mapper done in {mapper_time:.2f} s")

    # --------------------------
    # Image undistortion
    # --------------------------
    largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
    img_undist_cmd = [
        colmap_command, "image_undistorter",
        "--image_path", images_path,
        "--input_path", largest_sparse_folder,
        "--output_path", output_dir,
        "--output_type", "COLMAP"
    ]
    ret = run_subprocess(img_undist_cmd, logger)
    if ret != 0:
        logger.error("Image undistortion failed. Skipping to next data folder.")
    logger.info("Image undistortion done.")

    # --------------------------
    # Organize sparse output to sparse/0
    # --------------------------
    sparse_output_path = os.path.join(output_dir, "sparse")
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

    img_num, _, _ = count_images_in_dir(os.path.join(output_dir, "images"))

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