

def get_feature_extractor_cmd(colmap_command: str, 
                          log_level: str, 
                          database_path: str, 
                          images_path: str, 
                          images_list_path: str = "",
                          feature_type: str = "SIFT", # UNDEFINED, SIFT, ALIKED_N16ROT, ALIKED_N32
                          camera_mask_path: str = "",
                          image_mask_path: str = "",
                          single_camera_per_image: int = 0,
                          single_camera_per_fold: int = 0,
                          single_camera: int = 1,
                          camera_model: str = "SIMPLE_RADIAL",
                          default_focal_length_factor: float = 0.9,
                          camera_parameters: str = "",
                          use_gpu: int = 1,
                          max_image_size: int = 4096,
                          max_feature_num: int = 8192,
                          anms_selected_num: int = 1024,
                          cell_num: int = 600,
                          per_cell_num: int = 3,
                          sift_first_octave: int = -1, # -1
                          sift_num_octaves: int = 4, # 4
                          sift_octave_resolution: int = 3, # 3
                          sift_peak_threshold: float = 0.00667, # 0.00667
                          sift_edge_threshold: float = 10.0,    
                          aliked_n16rot_path: str = "models/aliked_n16rot.pt",
                          aliked_n32_path: str = "models/aliked_n32.pt",
                          loma_extractor_min_score: float = 0.0,
                          loma_extractor_use_bf16: int = 0,
                          loma_extractor_use_fast_resize: int = 0,
                          loma_detector_model_path: str = "models/loma_detector.onnx",
                          loma_descriptor_model_path: str = "models/loma_descriptor_dedode_g.onnx",
                          loma_descriptor_model_path_bf16: str = "models/loma_descriptor_dedode_g_bf16.onnx",
                          loma_descriptor_b128_model_path: str = "models/loma_descriptor_dedode_b.onnx"):
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--image_path", images_path,
        "--image_list_path", images_list_path,
        "--FeatureExtraction.type", feature_type, # UNDEFINED, SIFT, ALIKED_N16ROT, ALIKED_N32
        "--ImageReader.single_camera_per_image", str(single_camera_per_image),
        "--ImageReader.single_camera_per_fold", str(single_camera_per_fold),
        "--ImageReader.single_camera", str(single_camera),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.mask_path", image_mask_path,
        # "--ImageReader.existing_camera_id", str(args.existing_camera_id),
        "--ImageReader.camera_params", camera_parameters,
        "--ImageReader.default_focal_length_factor", str(default_focal_length_factor),
        "--ImageReader.camera_mask_path", camera_mask_path,
        # "--FeatureExtraction.num_threads", str(args.num_threads),
        "--FeatureExtraction.use_gpu", str(use_gpu),
        "--FeatureExtraction.gpu_index","-1",
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_feature_num),
        "--SiftExtraction.anms_selected_num", str(anms_selected_num),
        "--SiftExtraction.cell_num", str(cell_num),
        "--SiftExtraction.per_cell", str(per_cell_num),
        "--SiftExtraction.first_octave", str(sift_first_octave), # -1
        "--SiftExtraction.num_octaves", str(sift_num_octaves), # 4
        "--SiftExtraction.octave_resolution", str(sift_octave_resolution), # 3
        "--SiftExtraction.peak_threshold", str(sift_peak_threshold), # 0.00667
        "--SiftExtraction.edge_threshold", str(sift_edge_threshold), # 10
        # "--SiftExtraction.estimate_affine_shape", str(args.estimate_affine_shape),
        # "--SiftExtraction.max_num_orientations", str(args.max_num_orientations),
        # "--SiftExtraction.upright", str(args.upright),
        # "--SiftExtraction.domain_size_pooling", "0",
        # "--SiftExtraction.dsp_min_scale", "0.167",
        # "--SiftExtraction.dsp_max_scale", "3",
        # "--SiftExtraction.dsp_num_scales", "10",
        "--AlikedExtraction.max_num_features", str(max_feature_num),
        "--AlikedExtraction.min_score", "0.2", # 0.2
        "--AlikedExtraction.n16rot_model_path", aliked_n16rot_path,
        "--AlikedExtraction.n32_model_path", aliked_n32_path,
        "--LomaExtraction.max_num_features", str(max_feature_num),
        "--LomaExtraction.min_score", str(loma_extractor_min_score),
        "--LomaExtraction.use_bf16", str(loma_extractor_use_bf16),
        "--LomaExtraction.use_fast_resize", str(loma_extractor_use_fast_resize),
        "--LomaExtraction.detector_model_path", loma_detector_model_path,
        "--LomaExtraction.descriptor_model_path", loma_descriptor_model_path,
        "--LomaExtraction.descriptor_model_path_bf16", loma_descriptor_model_path_bf16,
        "--LomaExtraction.descriptor_b128_model_path", loma_descriptor_b128_model_path
    ]

    return feat_extraction_cmd

