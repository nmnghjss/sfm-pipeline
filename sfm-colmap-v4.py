import os
import sys
import logging
from argparse import ArgumentParser
import shutil
import time
from datetime import datetime
from mapper_cmd import get_ba_cmd, get_incremental_mapper_cmd, get_hierarchical_mapper_cmd, get_global_mapper_cmd, get_points_triangulate_cmd, get_pose_prior_global_mapper_cmd, get_pose_prior_mapper_cmd
from utils import run_subprocess, check_operating_system, count_images_in_dir_recursive, get_largest_subfolder, get_subfolders_names, clear_folder
from database import ReadColmapDatabase, filter_matches_by_inliers
from visualization import visualize_image_pairs

from match_utils import compute_matched_image_pairs_by_pose_prior
from database import initialize_colmap_database
from feature_extractor_cmd import get_feature_extractor_cmd, extract_neural_features
from match_cmd import get_exhaustive_matcher_cmd, generate_sequential_match_list, get_matches_importer_cmd, get_vocab_tree_matcher_cmd, match_features_with_lightglue

#  ========================== Argument parser ========================== 
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", default="E:\\Test1234\\data18", type=str)
parser.add_argument("--output_path", "-o", default="", type=str)
parser.add_argument("--camera", default="SIMPLE_RADIAL", type=str)
parser.add_argument("--default_focal_length_factor", default=1.0, type=float, help="Default focal length as a factor of image size (if not specified in EXIF)")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--single_camera", "-sc",default="1", type=str)
parser.add_argument("--single_fold", "-sf", default="0", type=str)
parser.add_argument("--single_image", "-si",default="0", type=str)
parser.add_argument("--feature_type", type=str, default="ALIKED_N16ROT", choices=["SIFT", "ALIKED_N16ROT", "ALIKED_N32"], help="Feature type for COLMAP feature extraction (e.g., SIFT, ALIKED_N16ROT, ALIKED_N32)")
parser.add_argument("--max_image_size", type=int, default=-1, help="maximum image size used to extract feature")
parser.add_argument("--match_strategy", "-ms", type=str, default="exhaustive", choices=["exhaustive", "sequential", "vocab_tree"], help="Matching strategy to use")
parser.add_argument("--match_alg", "-ma", type=str, default="LIGHTGLUE", choices=["BRUTEFORCE", "LIGHTGLUE"], help="Matching type for COLMAP (e.g., ALIKED_LIGHTGLUE, ALIKED_N32)")
parser.add_argument("--vocab_feature_num", type=int, default=0, help="vocab tree retrial feature num")
parser.add_argument("--mapper", default="pose_prior_global", type=str, choices=["incremental", "acc", "global", "hierarchical", "hierarchical_acc", "pose_prior", "pose_prior_global"], help="Algorithm for matching and mapping: colmap / acc / global / hierarchical / hierarchical_acc / pose_prior")
parser.add_argument("--max_feature_num", "-mfn", default=2048, type=int, help="Maximum number of features to extract per image (for SuperPoint)")
parser.add_argument("--min_num_inliers", type=int, default=30, help="Minimum number of inliers for a valid match")
parser.add_argument("--min_inlier_ratio", type=float, default=0.1, help="Minimum inlier ratio for a valid match")
parser.add_argument("--sequential_overlap", "-so", type=int, default=15, help="Number of neighboring images to match on each side for sequential matching")
parser.add_argument("--filt_match", action="store_true", help="Whether to filter matches by inliers before mapping")
parser.add_argument("--filter_inlier_ratio_threshold", type=float, default=0.6, help="Inlier ratio threshold for filtering matches before mapping")
parser.add_argument("--filter_inlier_num_threshold", type=int, default=30, help="Inlier number threshold for filtering matches before mapping")
parser.add_argument("--log_level", default="0", type=int, help="Set the logging level")
parser.add_argument("--visualize_matches", "-vis", action="store_true", help="Whether to visualize matches")
parser.add_argument("--clean", action="store_true", help="Whether to clean the output directory")
parser.add_argument("--external_splg", action="store_true", help="Whether to use external SuperPoint + LightGlue for feature extraction and matching instead of COLMAP's built-in methods")
parser.add_argument("--max_matches_per_image", "-mpi", type=int, default=30,
                    help="Max number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--min_matches_per_image", "-mni", type=int, default=10,
                    help="Minimum number of similar images to match per image (for nearest_k/quick strategies)")
parser.add_argument("--similarity_threshold", "-st", type=float, default=0.75,
                    help="Similarity threshold for threshold-based matching strategy (0~1)")
parser.add_argument("--ar_pose_path", type=str, help="Path to AR pose file (if available)")
parser.add_argument("--refine_num", type=int, default=3, help="refine reconstruction iterations num")
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
now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = os.path.join(log_dir, f"run-sfm-{now_str}.log")
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(log_level)
fh.setFormatter(log_formatter)
logger.addHandler(fh)


# ============================ Timeing variables ===============================
start_time = time.time()
feature_extraction_time = 0
feature_matching_time = 0
view_graph_calibrate_time = 0
triangulate_time = 0
ba_time = 0
refine_time = 0
mapper_time = 0

# =====================key parameter =====================================================
min_num_inliers = args.min_num_inliers
min_inlier_ratio = args.min_inlier_ratio

feature_match_type = args.match_alg
if feature_match_type != "UNDEFINED":
    if args.feature_type == "SIFT":
        feature_match_type = "SIFT" + "_" + feature_match_type
    else:
        feature_match_type = "ALIKED" + "_" + feature_match_type

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

bruteforce_match_path = os.path.join(current_path, "checkpoints/colmap_dep/bruteforce-matcher.onnx")
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
# print(f"First 5 images: {images_full_path[:5]}")
# ======================== GPU setup ==========================================
use_gpu = 0 if args.no_gpu else 1

# ============================ find matched pairs using slam pose (optional) ==========
prior_focal_length = None
if args.ar_pose_path is not None and len(args.ar_pose_path) > 0:
    logger.info(f"Finding image pairs based on AR poses from: {args.ar_pose_path}")
    # pcd=load_point_cloud(os.path.join(args.ar_pose_path, "scan_all_pointcloud.ply"))
    # poses=load_poses(os.path.join(args.ar_pose_path, "scan_all_pose.ply"))
    # intrinsics=load_intrinsics(os.path.join(args.ar_pose_path, "scan_camera_intrinsics.txt"))
    # build_frustum_overlap_voxel(
    #     pcd=pcd,
    #     poses=poses,
    #     intrinsics=intrinsics,
    #     output_txt=matched_images_pairs_path,
    #     voxel_size=0.1,
    #     max_pairs_per_image=args.max_matches_per_image,
    #     overlap_thresh=0.5,
    #     max_angle=np.radians(45)  # 45度视角约束
    # )

    if not os.path.isabs(args.ar_pose_path):
        args.ar_pose_path = os.path.join(args.source_path, args.ar_pose_path)
    prior_focal_length = compute_matched_image_pairs_by_pose_prior(args.ar_pose_path, matched_images_pairs_path, overlap_thresh=0.5)
    logger.info(f"AR pose-based image pairs saved to: {matched_images_pairs_path}, prior focal length: {prior_focal_length}")

# ========================= Feature extraction ==================================

if args.external_splg:
    logger.info("Using SuperPoint + LightGlue for feature extraction and matching...")
    logger.info(f"  Match strategy: {args.match_strategy}")
    if args.match_strategy in ["nearest_k", "quick"]:
        logger.info(f"  Max matches per image: {args.max_matches_per_image}")
    
    # Initialize COLMAP database with images (without SIFT extraction)
    logger.info("Initializing COLMAP database with image metadata (without feature extraction)...")
    splg_start = time.time()    
    db_init_success = initialize_colmap_database(
        database_path=database_path,
        images_dir= images_dir,
        images_path=images_full_path,
        camera_model=args.camera,
        prior_fx=prior_focal_length,
        prior_fy=prior_focal_length,
        logger=logger
    )    
    if not db_init_success:
        logger.error("Failed to initialize database!")
        sys.exit(1)
    
    if args.feature_type.lower().startswith("aliked"):
        logger.info(f"Using {args.feature_type} features for extraction and matching")
        args.feature_type = "aliked"
    # Now run SuperPoint feature extraction    
    image_features, image_id_map, feat_ext_time = extract_neural_features(
        feature_type=args.feature_type,
        local_weights_root=current_path,
        database_path=database_path,
        images_dir=images_dir,
        images_path=images_full_path,
        max_num_keypoints=args.max_feature_num,
        logger=logger
    )
    feature_extraction_time = time.time() - splg_start
    logger.info(f"SuperPoint feature extraction time: {feature_extraction_time:.2f} s (including database writes)")
    
    # Run LightGlue feature matching
    args.match_strategy = "threshold"
    match_start = time.time()
    feat_match_time = match_features_with_lightglue(
        current_path,
        args.feature_type,
        database_path,
        image_features,
        image_id_map,
        match_list_path=matched_images_pairs_path,
        match_strategy=args.match_strategy,
        max_matches_per_image=args.max_matches_per_image,
        min_matches_per_image=args.min_matches_per_image,
        similarity_threshold=args.similarity_threshold,
        logger=logger
    )

    # --- Import matches using COLMAP matches_importer ---
    logger.info("Importing matches into COLMAP database using matches_importer...")
    logger.info(f"Match list path: {matched_images_pairs_path}")
    logger.info(f"Database path: {database_path}")
    matches_importer_cmd = get_matches_importer_cmd(
        colmap_command=colmap_command,
        log_level = log_level, 
        database_path=database_path,
        matched_images_pairs_path=matched_images_pairs_path,
        feature_match_type = "SIFT_BRUTEFORCE", 
        use_gpu = 1,
        max_feature_num = 2048,
        min_num_inliers = 30,
        min_inlier_ratio = 0.1,                             
        sift_lightglue_match_path = sift_lightglue_match_path,
        bruteforce_match_path = bruteforce_match_path,
        aliked_lightglue_match_path = aliked_lightglue_match_path
    )
    run_subprocess(matches_importer_cmd, logger)
    feature_matching_time = time.time() - match_start
    logger.info(f"Matches imported successfully, match time: {feat_match_time}, match and import time: {feature_matching_time}")    

else:
    logger.info("Using COLMAP for feature extraction and matching...")
    feat_extraction_cmd = get_feature_extractor_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        feature_type=args.feature_type,
        single_camera_per_image=int(args.single_image),
        single_camera_per_fold=int(args.single_fold),
        single_camera=int(args.single_camera),
        camera_model=args.camera,
        default_focal_length_factor=args.default_focal_length_factor,
        use_gpu=use_gpu,
        max_image_size=args.max_image_size,
        max_feature_num=args.max_feature_num,
        aliked_n16rot_path=aliked_n16rot_path,
        aliked_n32_path=aliked_n32_path
    )
    logger.info("Starting feature extraction with COLMAP SIFT...")
    t0 = time.time()
    run_subprocess(feat_extraction_cmd, logger)
    feature_extraction_time = time.time() - t0
    logger.info(f"Feature extraction done in {feature_extraction_time:.2f} s")

    # ========================= Feature matching ========================= 
    logger.info("Starting feature matching...")
    t1 = time.time()
    if args.match_strategy == "exhaustive":
        feat_matching_cmd = get_exhaustive_matcher_cmd(
            colmap_command=colmap_command,
            log_level=log_level,
            database_path=database_path,
            feature_match_type=feature_match_type,
            use_gpu=use_gpu,
            max_feature_num=args.max_feature_num,
            min_num_inliers=min_num_inliers,
            min_inlier_ratio=min_inlier_ratio,
            sift_lightglue_match_path=sift_lightglue_match_path,
            bruteforce_match_path=bruteforce_match_path,
            aliked_lightglue_match_path=aliked_lightglue_match_path
        )
    elif args.match_strategy == "sequential":
        # Generate sequential match list with circular overlap
        logger.info("Generating sequential match pairs...")
        
        # Generate match pairs and save to file
        match_pairs = generate_sequential_match_list(
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
            max_feature_num=args.max_feature_num,
            min_num_inliers=min_num_inliers,
            min_inlier_ratio=min_inlier_ratio,
            sift_lightglue_match_path=sift_lightglue_match_path,
            bruteforce_match_path=bruteforce_match_path,
            aliked_lightglue_match_path=aliked_lightglue_match_path,
            max_matches_per_image=args.max_matches_per_image,
            vocab_feature_num=args.vocab_feature_num,
            vocab_path=vocab_path
        )
    run_subprocess(feat_matching_cmd, logger)
    feature_matching_time = time.time() - t1
    logger.info(f"Feature matching done in {feature_matching_time:.2f} s")

# ========================= Calibrate view graph ========================= 
view_graph_calibrate_start = time.time()
view_graph_calibrate_cmd = [
    colmap_command, "view_graph_calibrator",
    "--log_level", str(log_level),
    "--database_path", database_path,
    "--cross_validate_prior_focal_lengths", "1",
    "--min_calibrated_pair_ratio", "0.5",
    "--reestimate_relative_pose", "1",
    "--min_focal_length_ratio", "0.1",
    "--max_focal_length_ratio", "10",
    "--max_calibration_error", "2",
    "--relpose_max_error", "1",
    "--relpose_min_num_inliers", "30",
    "--relpose_min_inlier_ratio", "0.25",
]
run_subprocess(view_graph_calibrate_cmd, logger)
view_graph_calibrate_time = time.time() - view_graph_calibrate_start
logger.info(f"View graph calibrate done in {view_graph_calibrate_time:.2f} s")

# ========================= Visualize matches (optional) =========================
if args.visualize_matches:
    logger.info("Visualizing matches for a few image pairs...")
    view_graph, cameras, images, feature_name = ReadColmapDatabase(database_path)
    if view_graph is None or cameras is None or images is None:
        logger.warning("Could not read database for visualization, skipping visualization step")
    else:
        vis_dir = os.path.join(output_path, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)
        print("vis_dir: ", vis_dir)
        visualize_image_pairs(view_graph, images, images_dir, vis_dir, num_pairs=100)
        print("Visualization completed.")


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

# ========================= Points triangulation with pose prior (optional) =========================
if args.ar_pose_path is not None and len(args.ar_pose_path) and args.mapper == "pose_prior_global":
    logger.info("Using ar-based mapper...")

    triangulate_cmd = get_points_triangulate_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        ar_pose_path=args.ar_pose_path,
        min_num_inliers=min_num_inliers,
        tri_create_max_angle_error = 4.0, # 2.0
        tri_continue_max_angle_error = 4.0, # 2.0
        tri_merge_max_reproj_error = 8.0, # 4.0
        tri_complete_max_reproj_error = 8.0, # 4.0
        tri_complete_max_transitivity = 10, # 5
        tri_re_max_angle_error = 8.0 # 5       
    )
    triangulate_start = time.time()
    run_subprocess(triangulate_cmd, logger)
    triangulate_time = time.time() - triangulate_start
    logger.info(f"Triangulation completed in {triangulate_time:.2f} s")

    ba_start = time.time()
    ba_cmd = get_ba_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        input_path=distorted_sparse_path,
        output_path=distorted_sparse_path,
        refine_focal_length=1,
        refine_principal_point=0,
        refine_extra_params=1,
        refine_rig_from_world=1,
        refine_sensor_from_rig=1,
        refine_points3D=1,
        min_track_length=0,
        max_num_iterations=100,
        max_linear_solver_iterations=200,
        gradient_tolerance=0.0001,
        use_gpu=0
    )    
    run_subprocess(ba_cmd, logger)

    # run_subprocess(retriangulate_cmd, logger)
    # run_subprocess(ba_cmd, logger)

    ba_time = time.time() - ba_start
    logger.info(f"Bundle adjustment completed in {ba_time:.2f} s")
# ========================= Mapper / Bundle Adjustment =========================
mapper_log_path = os.path.join(output_path, "mapper.log")
logger.info("Starting mapper...")
t2 = time.time()
if args.mapper == "acc":
    mapper_cmd = get_incremental_mapper_cmd(
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
        abs_pose_min_inlier_ratio=min_inlier_ratio
    )
elif args.mapper == "hierarchical":
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
        abs_pose_min_inlier_ratio=min_inlier_ratio
    )
elif args.mapper == "global":
    mapper_cmd = get_global_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        min_num_inliers=min_num_inliers,
        ba_num_iterations=5,
        gp_max_num_iterations=100,
        ba_ceres_max_num_iterations=200
    )
elif args.mapper == "pose_prior_global":
    mapper_cmd = get_pose_prior_global_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        min_num_inliers=min_num_inliers,
        ba_num_iterations=5,
        gp_max_num_iterations=100,
        ba_ceres_max_num_iterations=200
    )
elif args.mapper == "pose_prior":
    mapper_cmd = get_pose_prior_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        input_path="",
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        abs_pose_min_num_inliers=min_num_inliers,
        abs_pose_min_inlier_ratio=min_inlier_ratio,
        ba_local_max_num_iterations = 25,
        ba_local_max_refinements = 2,
        ba_local_max_refinement_change = 0.001,
        ba_global_frames_ratio = 1.5,
        ba_global_points_ratio = 1.5,
        ba_global_max_num_iterations = 25,
        ba_global_max_refinements = 2,
        ba_global_max_refinement_change = 0.001,
        ba_global_frames_freq = 500,
        ba_global_points_freq = 250000,
        min_num_inliers = 30,
        min_inlier_ratio = 0.25,
        overwrite_priors_covariance  = 1,
        prior_position_std_x  = 1.0,
        prior_position_std_y  = 1.0,
        prior_position_std_z  = 1.0      
    )
else:
    mapper_cmd = get_incremental_mapper_cmd(
        colmap_command=colmap_command,
        log_level=log_level,
        database_path=database_path,
        images_path=images_dir,
        distorted_sparse_path=distorted_sparse_path,
        use_gpu=use_gpu,
        ba_local_max_num_iterations=25,
        ba_local_max_refinements=2,
        ba_local_max_refinement_change=0.001,
        ba_global_max_num_iterations=50,
        ba_global_max_refinements=5,
        ba_global_frames_ratio=1.1,
        ba_global_points_ratio=1.1,
        ba_global_max_refinement_change=0.0005,
        abs_pose_min_num_inliers=min_num_inliers,
        abs_pose_min_inlier_ratio=min_inlier_ratio
    )
mapper_start = time.time()
if args.mapper != "pose_prior_global":
    run_subprocess(mapper_cmd, logger)
else:
    if args.refine_num > 0:
        refine_start = time.time()
        for it in range(0, args.refine_num):
            logger.info(f" {it + 1} / 3 reconstruction refine")
            run_subprocess(mapper_cmd, logger)
        refine_time = time.time() - refine_start        
mapper_time = time.time() - mapper_start
mapper_time += triangulate_time + ba_time + view_graph_calibrate_time
logger.info(f"Mapper done in {mapper_time:.2f} s")

# ========================= Image undistortion ========================= 
statr_undistort = time.time()
input_subdirs = get_subfolders_names(images_dir)
for subdir in input_subdirs:
    output_subdir_path = os.path.join(output_path, "images", subdir)
    os.makedirs(output_subdir_path, exist_ok=True)
largest_sparse_folder = get_largest_subfolder(distorted_sparse_path)
if args.refine_num == 0 and args.mapper == "pose_prior_global":
    largest_sparse_folder = distorted_sparse_path
logger.info(f"largest_sparse_folder: {largest_sparse_folder}")
img_undist_cmd = [
    colmap_command, "image_undistorter",
    "--image_path", images_dir,
    "--input_path", largest_sparse_folder,
    "--output_path", output_path,
    "--output_type", "COLMAP"
]
run_subprocess(img_undist_cmd, logger)
undistort_time = time.time() - statr_undistort
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
img_num, _ = count_images_in_dir_recursive(os.path.join(output_path, "images"))
logger.info("Sparse output successfully organized into sparse/0.")

# --------------------------
# Timing summary
# --------------------------
total_time = time.time() - start_time
sparse_reconstruction_time = feature_extraction_time + feature_matching_time + mapper_time
# pair_match_time = feature_matching_time / (input_img_num * (input_img_num - 1) / 2) if input_img_num > 1 else 0

logger.info("Done. Timing statistics:")
logger.info(f"  Feature extraction: {int(feature_extraction_time // 60)} min {feature_extraction_time % 60:.2f} s")
logger.info(f"  Feature matching: {int(feature_matching_time // 60)} min {feature_matching_time % 60:.2f} s")
# logger.info(f"  Average time per image pair: {pair_match_time * 1000:.2f} ms")
logger.info(f"  View Grpha Calibrate time: {int(view_graph_calibrate_time // 60)} min {view_graph_calibrate_time % 60:.2f} s")
logger.info(f"  Triangulation time: {int(triangulate_time // 60)} min {triangulate_time % 60:.2f} s")
logger.info(f"  Bundle Adjustment time: {int(ba_time // 60)} min {ba_time % 60:.2f} s")
logger.info(f"  Refine time: {int(refine_time // 60)} min {refine_time % 60:.2f} s")
logger.info(f"  Mapper: {int(mapper_time // 60)} min {mapper_time % 60:.2f} s")
logger.info(f"  Sparse reconstruction: {int(sparse_reconstruction_time // 60)} min {sparse_reconstruction_time % 60:.2f} s")
logger.info(f"  Image undistortion: {int(undistort_time // 60)} min {undistort_time % 60:.2f} s")
logger.info(f"  Total time: {int(total_time // 60)} min {total_time % 60:.2f} s")
logger.info(f"  Number of images registered: {img_num} / {input_img_num}")