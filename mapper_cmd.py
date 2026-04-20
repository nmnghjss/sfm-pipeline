

def get_incremental_mapper_cmd(colmap_command: str, 
                             log_level: int, 
                             database_path: str, 
                             images_path: str,
                             distorted_sparse_path: str,
                             min_num_matches: int = 15, 
                             init_num_trials: int = 200,
                             init_min_num_inliers: int = 100,
                             init_max_error: float = 4.0,
                             use_gpu: int = 0,
                             ba_local_max_num_iterations: int = 25,
                             ba_local_max_refinements: int = 2,
                             ba_local_max_refinement_change: float = 0.001,
                             ba_global_max_num_iterations: int = 25,
                             ba_global_max_refinements: int = 5,
                             ba_global_max_refinement_change: float = 0.0005,
                             ba_global_frames_ratio: float = 1.1,
                             ba_global_points_ratio: float = 1.1,
                             ba_global_frames_freq: int = 500,
                             ba_global_points_freq: int = 250000,
                             abs_pose_min_num_inliers: int = 30,
                             abs_pose_min_inlier_ratio: float = 0.25):

    mapper_cmd = [
        colmap_command, "mapper",
        "--log_level", str(log_level),        
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.num_threads", "-1",
        "--Mapper.min_num_matches", str(min_num_matches), # 15
        "--Mapper.init_num_trials", str(init_num_trials), # 200
        "--Mapper.init_min_num_inliers", str(init_min_num_inliers), # 100
        "--Mapper.init_max_error", str(init_max_error), # 4
        "--Mapper.init_min_tri_angle", "16", # 16
        "--Mapper.ba_local_min_tri_angle", "6", # 6
        "--Mapper.ba_local_num_images", "6", # 6
        "--Mapper.ba_local_max_num_iterations", str(ba_local_max_num_iterations), # 25
        "--Mapper.ba_local_max_refinements", str(ba_local_max_refinements), # 2
        "--Mapper.ba_local_max_refinement_change", str(ba_local_max_refinement_change), # 0.001
        "--Mapper.ba_global_frames_ratio", str(ba_global_frames_ratio), # 1.1
        "--Mapper.ba_global_points_ratio", str(ba_global_points_ratio), # 1.1
        "--Mapper.ba_global_frames_freq", str(ba_global_frames_freq), # 500
        "--Mapper.ba_global_points_freq", str(ba_global_points_freq), # 250000
        "--Mapper.ba_global_max_num_iterations", str(ba_global_max_num_iterations), # 50 --> 20 --> 25
        "--Mapper.ba_global_max_refinements", str(ba_global_max_refinements), # 5
        "--Mapper.ba_global_max_refinement_change", str(ba_global_max_refinement_change), # 0.0005
        "--Mapper.ba_refine_focal_length", "1", # 1
        "--Mapper.ba_refine_principal_point", "0", # 0
        "--Mapper.ba_refine_extra_params", "1", # 1
        "--Mapper.ba_use_gpu", str(use_gpu),  # 0
        "--Mapper.abs_pose_max_error", "12", # 12
        "--Mapper.abs_pose_min_num_inliers", str(abs_pose_min_num_inliers), # 30
        "--Mapper.abs_pose_min_inlier_ratio", str(abs_pose_min_inlier_ratio), # 0.25
        "--Mapper.max_extra_param", "1", # 1
        "--Mapper.tri_min_angle", "1.5", # 1.5
        "--Mapper.tri_create_max_angle_error", "2", # 2
        "--Mapper.tri_merge_max_reproj_error", "4", # 4
        "--Mapper.filter_max_reproj_error", "4", # 4
        "--Mapper.max_reg_trials", "3", # 3
    ]

    return mapper_cmd


def get_hierarchical_mapper_cmd(colmap_command: str,
                             log_level: int, 
                             database_path: str, 
                             images_path: str,
                             distorted_sparse_path: str,
                             use_gpu: int = 0,
                             leaf_max_num_images: int = 500,
                             init_num_trials: int = 1000,
                             ba_local_max_num_iterations: int = 25,
                             ba_local_max_refinements: int = 2,
                             ba_local_max_refinement_change: float = 0.001,
                             ba_global_frames_ratio: float = 1.1,
                             ba_global_points_ratio: float = 1.1,
                             ba_global_frames_freq: int = 500,
                             ba_global_points_freq: int = 250000,
                             ba_global_max_num_iterations: int = 50,
                             ba_global_max_refinements: int = 5,
                             ba_global_max_refinement_change: float = 0.0005,
                             abs_pose_min_num_inliers: int = 30,
                             abs_pose_min_inlier_ratio: float = 0.25):

    mapper_cmd = [
        colmap_command, "hierarchical_mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--log_level", str(log_level),
        "--num_workers", "-1",
        "--image_overlap", "50", # 50
        "--leaf_max_num_images", str(leaf_max_num_images), # 500
        "--Mapper.min_num_matches", "15",
        "--Mapper.ignore_watermarks", "0",
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "50",
        "--Mapper.max_model_overlap", "20",
        "--Mapper.min_model_size", "10",
        "--Mapper.init_num_trials", str(init_num_trials), # 200
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", "-1",
        "--Mapper.random_seed", "-1",
        "--Mapper.min_focal_length_ratio", "0.1",
        "--Mapper.max_focal_length_ratio", "10",
        "--Mapper.max_extra_param", "1",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_refine_sensor_from_rig", "0",
        "--Mapper.ba_local_function_tolerance", "0",
        "--Mapper.ba_local_max_num_iterations", str(ba_local_max_num_iterations), #25
        "--Mapper.ba_local_max_refinements", str(ba_local_max_refinements), # 2
        "--Mapper.ba_local_max_refinement_change", str(ba_local_max_refinement_change), # 0.001
        "--Mapper.ba_global_frames_ratio", str(ba_global_frames_ratio), # 1.1
        "--Mapper.ba_global_points_ratio", str(ba_global_points_ratio), # 1.1
        "--Mapper.ba_global_frames_freq", str(ba_global_frames_freq), # 500
        "--Mapper.ba_global_points_freq", str(ba_global_points_freq), # 250000
        "--Mapper.ba_global_function_tolerance", "0",
        "--Mapper.ba_global_max_num_iterations", str(ba_global_max_num_iterations), # 50
        "--Mapper.ba_global_max_refinements", str(ba_global_max_refinements), # 5
        "--Mapper.ba_global_max_refinement_change", str(ba_global_max_refinement_change),
        "--Mapper.ba_use_gpu", str(use_gpu),
        "--Mapper.ba_gpu_index", "-1",
        "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        "--Mapper.snapshot_path", "",
        "--Mapper.snapshot_frames_freq", "0",
        "--Mapper.fix_existing_frames", "0",
        "--Mapper.init_min_num_inliers", "100",
        "--Mapper.init_max_error", "4",
        "--Mapper.init_max_forward_motion", "0.95",
        "--Mapper.init_min_tri_angle", "16",
        "--Mapper.init_max_reg_trials", "2",
        "--Mapper.abs_pose_max_error", "12",
        "--Mapper.abs_pose_min_num_inliers", str(abs_pose_min_num_inliers),
        "--Mapper.abs_pose_min_inlier_ratio", str(abs_pose_min_inlier_ratio),
        "--Mapper.filter_max_reproj_error", "4",
        "--Mapper.filter_min_tri_angle", "1.5",
        "--Mapper.max_reg_trials", "3",
        "--Mapper.ba_local_num_images", "6",
        "--Mapper.ba_local_min_tri_angle", "6",
        "--Mapper.ba_global_ignore_redundant_points3D", "0",
        "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", "0.05",
        "--Mapper.image_list_path", "",
        "--Mapper.constant_rig_list_path", "",
        "--Mapper.constant_camera_list_path", "",
        "--Mapper.max_runtime_seconds", "-1",
        "--Mapper.tri_max_transitivity", "1",
        "--Mapper.tri_create_max_angle_error", "2",
        "--Mapper.tri_continue_max_angle_error", "2",
        "--Mapper.tri_merge_max_reproj_error", "4",
        "--Mapper.tri_complete_max_reproj_error", "4",
        "--Mapper.tri_complete_max_transitivity", "5",
        "--Mapper.tri_re_max_angle_error", "5",
        "--Mapper.tri_re_min_ratio", "0.2",
        "--Mapper.tri_re_max_trials", "1",
        "--Mapper.tri_min_angle", "1.5",
        "--Mapper.tri_ignore_two_view_tracks", "1"
    ]

    return mapper_cmd


def get_global_mapper_cmd(colmap_command: str, 
                   log_level: int, 
                   database_path: str, 
                   images_path: str,
                   distorted_sparse_path: str,
                   use_gpu: int = 0,
                   min_num_inliers: int = 30,
                   ba_num_iterations: int = 3,
                   gp_max_num_iterations: int = 100,
                   ba_ceres_max_num_iterations: int = 200):

    mapper_cmd = [
        colmap_command, "global_mapper",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--GlobalMapper.image_list_path", "",
        "--GlobalMapper.min_num_matches", str(min_num_inliers),
        "--GlobalMapper.ignore_watermarks", "0",
        "--GlobalMapper.num_threads", "-1",
        "--GlobalMapper.random_seed", "-1",
        "--GlobalMapper.decompose_relative_pose", "1",
        "--GlobalMapper.ba_num_iterations", str(ba_num_iterations),
        "--GlobalMapper.skip_rotation_averaging", "0",
        "--GlobalMapper.skip_track_establishment", "0",
        "--GlobalMapper.skip_global_positioning", "0",
        "--GlobalMapper.skip_bundle_adjustment", "0",
        "--GlobalMapper.skip_retriangulation", "0",
        "--GlobalMapper.track_intra_image_consistency_threshold", "10",
        "--GlobalMapper.track_required_tracks_per_view", "2147483647",
        "--GlobalMapper.track_min_num_views_per_track", "3",
        "--GlobalMapper.gp_use_gpu", str(use_gpu), # 1
        "--GlobalMapper.gp_gpu_index", "-1",
        "--GlobalMapper.gp_optimize_positions", "1",
        "--GlobalMapper.gp_optimize_points", "1",
        "--GlobalMapper.gp_optimize_scales", "1",
        "--GlobalMapper.gp_loss_function_scale", "0.1",
        "--GlobalMapper.gp_max_num_iterations", str(gp_max_num_iterations), # 100
        "--GlobalMapper.ba_refine_focal_length", "1",
        "--GlobalMapper.ba_refine_principal_point", "0",
        "--GlobalMapper.ba_refine_extra_params", "1",
        "--GlobalMapper.ba_refine_sensor_from_rig", "0",
        "--GlobalMapper.ba_refine_rig_from_world", "1",
        "--GlobalMapper.ba_refine_points3D", "1",
        "--GlobalMapper.ba_min_track_length", "3",
        "--GlobalMapper.ba_ceres_use_gpu", "0", # 1
        "--GlobalMapper.ba_ceres_gpu_index", "-1",
        "--GlobalMapper.ba_ceres_loss_function_scale", "1",
        "--GlobalMapper.ba_ceres_max_num_iterations", str(ba_ceres_max_num_iterations),
        "--GlobalMapper.ba_skip_fixed_rotation_stage", "0",
        "--GlobalMapper.ba_skip_joint_optimization_stage", "0",
        "--GlobalMapper.tri_complete_max_reproj_error", "15",
        "--GlobalMapper.tri_merge_max_reproj_error", "15",
        "--GlobalMapper.tri_min_angle", "1", #1
        "--GlobalMapper.ra_max_rotation_error_deg", "10",
        "--GlobalMapper.max_angular_reproj_error_deg", "1",
        "--GlobalMapper.max_normalized_reproj_error", "0.01",
        "--GlobalMapper.min_tri_angle_deg", "1", #1
    ]
    return mapper_cmd


def get_pose_prior_mapper_cmd(colmap_command: str, 
                             log_level: int, 
                             database_path: str, 
                             images_path: str,
                             input_path: str,
                             distorted_sparse_path: str,
                             use_gpu: int = 0,
                             ba_local_max_num_iterations: int = 25,
                             ba_local_max_refinements: int = 2,
                             ba_local_max_refinement_change: float = 0.001,
                             ba_global_frames_ratio: float = 1.5,
                             ba_global_points_ratio: float = 1.5,
                             ba_global_max_num_iterations: int = 25,
                             ba_global_max_refinements: int = 2,
                             ba_global_max_refinement_change: float = 0.001,
                             ba_global_frames_freq: int = 500,
                             ba_global_points_freq: int = 250000,
                             min_num_inliers: int = 30,
                             min_inlier_ratio: float = 0.25,
                             overwrite_priors_covariance: int = 1,
                             prior_position_std_x: float = 1.0,
                             prior_position_std_y: float = 1.0,
                             prior_position_std_z: float = 1.0,):

    mapper_cmd = [
        colmap_command, "pose_prior_mapper",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--image_path", images_path,
        "--input_path", input_path,
        "--output_path", distorted_sparse_path,
        "--Mapper.min_num_matches", str(min_num_inliers),
        "--Mapper.ignore_watermarks", "0",
        "--Mapper.extract_colors", "1",
        "--Mapper.num_threads", "-1",
        "--Mapper.random_seed", "-1",
        "--Mapper.ba_use_gpu", str(use_gpu), # 0
        "--Mapper.ba_gpu_index", "-1",        
        "--Mapper.min_focal_length_ratio", "0.1",
        "--Mapper.max_focal_length_ratio", "10",
        "--Mapper.max_extra_param", "1",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_refine_sensor_from_rig", "1",
        "--Mapper.ba_local_num_images", "6", # 6
        "--Mapper.ba_local_min_tri_angle", "6",        
        "--Mapper.ba_local_function_tolerance", "0",
        "--Mapper.ba_local_max_num_iterations", str(ba_local_max_num_iterations),
        "--Mapper.ba_local_max_refinements", str(ba_local_max_refinements),
        "--Mapper.ba_local_max_refinement_change", str(ba_local_max_refinement_change),
        "--Mapper.ba_global_frames_ratio", str(ba_global_frames_ratio), # 1.1
        "--Mapper.ba_global_points_ratio", str(ba_global_points_ratio), # 1.1
        "--Mapper.ba_global_frames_freq", str(ba_global_frames_freq), # 500
        "--Mapper.ba_global_points_freq", str(ba_global_points_freq), # 250000
        "--Mapper.ba_global_function_tolerance", "0",
        "--Mapper.ba_global_max_num_iterations", str(ba_global_max_num_iterations), # 50
        "--Mapper.ba_global_max_refinements", str(ba_global_max_refinements), # 5
        "--Mapper.ba_global_max_refinement_change", str(ba_global_max_refinement_change), # 0.0005
        "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        "--Mapper.snapshot_path", "",
        "--Mapper.snapshot_frames_freq", "0",
        "--Mapper.fix_existing_frames", "0",
        "--Mapper.init_min_num_inliers", "1000", # 100
        "--Mapper.init_max_error", "4",
        "--Mapper.init_max_forward_motion", "0.95",
        "--Mapper.init_min_tri_angle", "16",
        "--Mapper.init_max_reg_trials", "2",
        "--Mapper.abs_pose_max_error", "12",
        "--Mapper.abs_pose_min_num_inliers", str(min_num_inliers), # 30
        "--Mapper.abs_pose_min_inlier_ratio", str(min_inlier_ratio), # 0.25
        "--Mapper.filter_max_reproj_error", "4",
        "--Mapper.filter_min_tri_angle", "1.5",
        "--Mapper.max_reg_trials", "3",
        "--Mapper.ba_global_ignore_redundant_points3D", "0",
        "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", "0.05",
        "--Mapper.image_list_path", "",
        "--Mapper.constant_rig_list_path", "",
        "--Mapper.constant_camera_list_path", "",
        "--Mapper.max_runtime_seconds", "-1",
        "--Mapper.tri_max_transitivity", "1",
        "--Mapper.tri_create_max_angle_error", "2",
        "--Mapper.tri_continue_max_angle_error", "2",
        "--Mapper.tri_merge_max_reproj_error", "4",
        "--Mapper.tri_complete_max_reproj_error", "4",
        "--Mapper.tri_complete_max_transitivity", "5",
        "--Mapper.tri_re_max_angle_error", "5",
        "--Mapper.tri_re_min_ratio", "0.2",
        "--Mapper.tri_re_max_trials", "1",
        "--Mapper.tri_min_angle", "1.5",
        "--Mapper.tri_ignore_two_view_tracks", "1",
        "--overwrite_priors_covariance", str(overwrite_priors_covariance),
        "--prior_position_std_x", str(prior_position_std_x),
        "--prior_position_std_y", str(prior_position_std_y),
        "--prior_position_std_z", str(prior_position_std_z),
        "--use_robust_loss_on_prior_position", "0",
        "--prior_position_loss_scale", "7.82",
    ]

    return mapper_cmd


def get_points_triangulate_cmd(colmap_command: str,
                             log_level: int,
                             database_path: str,
                             images_path: str,
                             distorted_sparse_path: str,
                             ar_pose_path: str,
                             min_num_inliers: int = 30):

    triangulate_cmd = [
        colmap_command, "point_triangulator",
        "--database_path", database_path,
        "--image_path", images_path,
        "--input_path", ar_pose_path,
        "--output_path", distorted_sparse_path,
        "--log_level", str(log_level),
        "--clear_points", "1",              
        "--refine_intrinsics", "0", # 0
        "--Mapper.min_num_matches", str(min_num_inliers), # 15
        # "--Mapper.ignore_watermarks", "1", # 0
        # "--Mapper.multiple_models", "1",
        # "--Mapper.max_num_models", "50",
        # "--Mapper.max_model_overlap", "20",
        # "--Mapper.min_model_size", "10",
        # "--Mapper.init_image_id1", "-1",
        # "--Mapper.init_image_id2", "-1",
        # "--Mapper.init_num_trials", "1000", # 200
        # "--Mapper.structure_less_registration_fallback", "1",
        # "--Mapper.structure_less_registration_only", "0",
        # "--Mapper.extract_colors", "1",
        # "--Mapper.num_threads", "-1",
        # "--Mapper.random_seed", "-1",
        # "--Mapper.min_focal_length_ratio", "0.1",
        # "--Mapper.max_focal_length_ratio", "10",
        # "--Mapper.max_extra_param", "1",
        # "--Mapper.ba_refine_focal_length", "1",
        # "--Mapper.ba_refine_principal_point", "0",
        # "--Mapper.ba_refine_extra_params", "1",
        # "--Mapper.ba_refine_sensor_from_rig", "1",
        # "--Mapper.ba_local_function_tolerance", "0",
        # "--Mapper.ba_local_max_num_iterations", "25",
        # "--Mapper.ba_global_frames_ratio", "1.1",
        # "--Mapper.ba_global_points_ratio", "1.1",
        # "--Mapper.ba_global_frames_freq", "500",
        # "--Mapper.ba_global_points_freq", "250000",
        # "--Mapper.ba_global_function_tolerance", "0",
        # "--Mapper.ba_global_max_num_iterations", "50",
        # "--Mapper.ba_global_max_refinements", "5",
        # "--Mapper.ba_global_max_refinement_change", "0.0005",
        # "--Mapper.ba_local_max_refinements", "2",
        # "--Mapper.ba_local_max_refinement_change", "0.001",
        # "--Mapper.ba_use_gpu", "0",
        # "--Mapper.ba_gpu_index", "-1",
        # "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "50000",
        # "--Mapper.snapshot_path", "snapshot",
        # "--Mapper.snapshot_frames_freq", "0",
        # "--Mapper.fix_existing_frames", "0",
        # "--Mapper.init_min_num_inliers", "100",
        # "--Mapper.init_max_error", "4",
        # "--Mapper.init_max_forward_motion", "0.95",
        # "--Mapper.init_min_tri_angle", "16",
        # "--Mapper.init_max_reg_trials", "2",
        # "--Mapper.abs_pose_max_error", "12",
        # "--Mapper.abs_pose_min_num_inliers", "30",
        # "--Mapper.abs_pose_min_inlier_ratio", "0.25",
        # "--Mapper.filter_max_reproj_error", "4",
        # "--Mapper.filter_min_tri_angle", "1.5",
        # "--Mapper.max_reg_trials", "3",
        # "--Mapper.ba_local_num_images", "6",
        # "--Mapper.ba_local_min_tri_angle", "6",
        # "--Mapper.ba_global_ignore_redundant_points3D", "0",
        # "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", "0.05",
        # "--Mapper.image_list_path",
        # "--Mapper.constant_rig_list_path",
        # "--Mapper.constant_camera_list_path",
        # "--Mapper.max_runtime_seconds", "-1",
        # "--Mapper.tri_max_transitivity", "1",
        # "--Mapper.tri_create_max_angle_error", "2",
        # "--Mapper.tri_continue_max_angle_error", "2",
        # "--Mapper.tri_merge_max_reproj_error", "4",
        # "--Mapper.tri_complete_max_reproj_error", "4",
        # "--Mapper.tri_complete_max_transitivity", "5",
        # "--Mapper.tri_re_max_angle_error", "5",
        # "--Mapper.tri_re_min_ratio", "0.2",
        # "--Mapper.tri_re_max_trials", "1",
        # "--Mapper.tri_min_angle", "1.5",
        # "--Mapper.tri_ignore_two_view_tracks", "1"
    ]
    return triangulate_cmd
    

def get_ba_cmd(colmap_command: str,
               log_level: int,
               input_path: str,
               output_path: str,
               refine_focal_length: int = 1,
               refine_principal_point: int = 0,
               refine_extra_params: int = 1,
               refine_rig_from_world: int = 1,
               refine_sensor_from_rig: int = 1,
               refine_points3D: int = 1,
               min_track_length: int = 0,
               max_num_iterations: int = 100,
               max_linear_solver_iterations: int = 200,
               gradient_tolerance: float = 0.0001,
               use_gpu: int = 0):

    ba_cmd = [
        colmap_command, "bundle_adjuster",
        "--input_path", input_path,
        "--output_path", output_path,
        "--log_level", str(log_level),
        "--BundleAdjustment.refine_focal_length", str(refine_focal_length),
        "--BundleAdjustment.refine_principal_point", str(refine_principal_point),
        "--BundleAdjustment.refine_extra_params", str(refine_extra_params),
        "--BundleAdjustment.refine_rig_from_world", str(refine_rig_from_world),
        "--BundleAdjustment.refine_sensor_from_rig", str(refine_sensor_from_rig),
        "--BundleAdjustment.refine_points3D", str(refine_points3D),
        "--BundleAdjustment.constant_rig_from_world_rotation", "0",
        "--BundleAdjustment.min_track_length", str(min_track_length),
        "--BundleAdjustmentCeres.max_num_iterations", str(max_num_iterations),
        "--BundleAdjustmentCeres.max_linear_solver_iterations", str(max_linear_solver_iterations),
        "--BundleAdjustmentCeres.function_tolerance", "0",
        "--BundleAdjustmentCeres.gradient_tolerance", str(gradient_tolerance),
        "--BundleAdjustmentCeres.parameter_tolerance", "0",
        "--BundleAdjustmentCeres.use_gpu", str(use_gpu),
        "--BundleAdjustmentCeres.gpu_index", "-1",
        "--BundleAdjustmentCeres.min_num_images_gpu_solver", "50",
        "--BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading", "50000",
        "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver", "50",
        "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver", "1000",
        "--BundleAdjustmentCeres.max_num_images_direct_dense_gpu_solver", "200",
        "--BundleAdjustmentCeres.max_num_images_direct_sparse_gpu_solver", "4000"
    ]
    return ba_cmd



def get_pose_prior_global_mapper_cmd(colmap_command: str, 
                   log_level: int, 
                   database_path: str, 
                   images_path: str,
                   distorted_sparse_path: str,
                   use_gpu: int = 0,
                   min_num_inliers: int = 30,
                   ba_num_iterations: int = 3,
                   gp_max_num_iterations: int = 100,
                   ba_ceres_max_num_iterations: int = 200,
                   ba_skip_fixed_points_stage: int = 1,
                   ba_skip_fixed_rotation_stage: int = 1,
                   ba_skip_joint_optimization_stage: int = 1,
                   max_angular_reproj_error_deg: float = 1.0,
                   max_normalized_reproj_error: float = 0.01,
                   min_tri_angle_deg: float = 1.0
                   ):

    mapper_cmd = [
        colmap_command, "pose_prior_global_mapper",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", distorted_sparse_path,
        "--prior_reconstruction_path", distorted_sparse_path,
        "--GlobalMapper.image_list_path", "",
        "--GlobalMapper.min_num_matches", str(min_num_inliers),
        "--GlobalMapper.ignore_watermarks", "0",
        "--GlobalMapper.num_threads", "-1",
        "--GlobalMapper.random_seed", "-1",
        "--GlobalMapper.decompose_relative_pose", "1",
        "--GlobalMapper.ba_num_iterations", str(ba_num_iterations),
        "--GlobalMapper.skip_rotation_averaging", "1",
        "--GlobalMapper.skip_track_establishment", "1",
        "--GlobalMapper.skip_global_positioning", "1",
        "--GlobalMapper.skip_bundle_adjustment", "0",
        "--GlobalMapper.skip_retriangulation", "0",
        "--GlobalMapper.track_intra_image_consistency_threshold", "10",
        "--GlobalMapper.track_required_tracks_per_view", "2147483647",
        "--GlobalMapper.track_min_num_views_per_track", "3",
        "--GlobalMapper.gp_use_gpu", str(use_gpu), # 1
        "--GlobalMapper.gp_gpu_index", "-1",
        "--GlobalMapper.gp_optimize_positions", "1",
        "--GlobalMapper.gp_optimize_points", "1",
        "--GlobalMapper.gp_optimize_scales", "1",
        "--GlobalMapper.gp_loss_function_scale", "0.1",
        "--GlobalMapper.gp_max_num_iterations", str(gp_max_num_iterations), # 100
        "--GlobalMapper.ba_refine_focal_length", "1",
        "--GlobalMapper.ba_refine_principal_point", "0",
        "--GlobalMapper.ba_refine_extra_params", "1",
        "--GlobalMapper.ba_refine_sensor_from_rig", "0",
        "--GlobalMapper.ba_refine_rig_from_world", "1",
        "--GlobalMapper.ba_refine_points3D", "1",
        "--GlobalMapper.ba_min_track_length", "3",
        "--GlobalMapper.ba_ceres_use_gpu", "0", # 1
        "--GlobalMapper.ba_ceres_gpu_index", "-1",
        "--GlobalMapper.ba_ceres_loss_function_scale", "1",
        "--GlobalMapper.ba_ceres_max_num_iterations", str(ba_ceres_max_num_iterations),
        "--GlobalMapper.ba_skip_fixed_points_stage", str(ba_skip_fixed_points_stage),        
        "--GlobalMapper.ba_skip_fixed_rotation_stage", str(ba_skip_fixed_rotation_stage),
        "--GlobalMapper.ba_skip_joint_optimization_stage", str(ba_skip_joint_optimization_stage),
        "--GlobalMapper.tri_complete_max_reproj_error", "15",
        "--GlobalMapper.tri_merge_max_reproj_error", "15",
        "--GlobalMapper.tri_min_angle", "1", #1
        "--GlobalMapper.ra_max_rotation_error_deg", "10",
        "--GlobalMapper.max_angular_reproj_error_deg", str(max_angular_reproj_error_deg), # 1
        "--GlobalMapper.max_normalized_reproj_error", str(max_normalized_reproj_error), # 0.01
        "--GlobalMapper.min_tri_angle_deg", str(min_tri_angle_deg), #1
    ]
    return mapper_cmd