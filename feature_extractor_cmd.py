import os
import time
import logging
import torch
from lightglue import ALIKED, SuperPoint, DISK
from database import COLMAPDatabase, batch_write_keypoints_to_database
from lightglue.utils import load_image_use_torchvision
import numpy as np

def get_feature_extractor_cmd(colmap_command: str, 
                          log_level: str, 
                          database_path: str, 
                          images_path: str, 
                          feature_type: str,
                          single_camera_per_image: int = 0,
                          single_camera_per_fold: int = 0,
                          single_camera: int = 1,
                          camera_model: str = "SIMPLE_RADIAL",
                          default_focal_length_factor: float = 0.9,
                          use_gpu: int = 1,
                          max_image_size: int = 4096,
                          max_feature_num: int = 8192,
                          aliked_n16rot_path: str = "models/aliked_n16rot.pt",
                          aliked_n32_path: str = "models/aliked_n32.pt"):
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--log_level", str(log_level),
        "--database_path", database_path,
        "--image_path", images_path,
        "--FeatureExtraction.type", feature_type, # UNDEFINED, SIFT, ALIKED_N16ROT, ALIKED_N32
        "--ImageReader.single_camera_per_image", str(single_camera_per_image),
        "--ImageReader.single_camera_per_fold", str(single_camera_per_fold),
        "--ImageReader.single_camera", str(single_camera),
        "--ImageReader.camera_model", camera_model,
        # "--ImageReader.mask_path", args.mask_path,
        # "--ImageReader.existing_camera_id", str(args.existing_camera_id),
        # "--ImageReader.camera_params", args.camera_params,
        "--ImageReader.default_focal_length_factor", str(default_focal_length_factor),
        # "--ImageReader.camera_mask_path", args.camera_mask_path,
        # "--FeatureExtraction.num_threads", str(args.num_threads),
        "--FeatureExtraction.use_gpu", str(use_gpu),
        "--FeatureExtraction.gpu_index","-1",
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_feature_num),
        # "--SiftExtraction.first_octave", str(args.first_octave),
        # "--SiftExtraction.num_octaves", str(args.num_octaves),
        # "--SiftExtraction.octave_resolution", str(args.octave_resolution),
        # "--SiftExtraction.peak_threshold", str(args.peak_threshold),
        # "--SiftExtraction.edge_threshold", str(args.edge_threshold),
        # "--SiftExtraction.estimate_affine_shape", str(args.estimate_affine_shape),
        # "--SiftExtraction.max_num_orientations", str(args.max_num_orientations),
        # "--SiftExtraction.upright", str(args.upright),
        "--SiftExtraction.domain_size_pooling", "0",
        "--SiftExtraction.dsp_min_scale", "0.167",
        "--SiftExtraction.dsp_max_scale", "3",
        "--SiftExtraction.dsp_num_scales", "10",
        "--AlikedExtraction.max_num_features", str(max_feature_num),
        "--AlikedExtraction.min_score", "0.2",
        "--AlikedExtraction.n16rot_model_path", aliked_n16rot_path,
        "--AlikedExtraction.n32_model_path", aliked_n32_path
    ]

    return feat_extraction_cmd


def extract_neural_features(
    feature_type: str,
    local_weights_root: str,
    database_path: str,
    images_dir: str,
    images_path: str,
    max_num_keypoints: int = 2048,
    logger: logging.Logger = None
) -> tuple:
    """
    Extract features using SuperPoint and write to COLMAP database.
    
    Args:
        database_path: Path to COLMAP database
        images_path: Path to images directory
        max_num_keypoints: Maximum number of keypoints per image
        logger: Optional logger instance
    
    Returns:
        (image_features, image_id_map, feature_extraction_time)
        - image_features: Dict of image name -> feature tensor
        - image_id_map: Dict of image name -> image_id in database
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Initialize SuperPoint model
    logger.info("Initializing SuperPoint model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    # local_weights_root = "checkpoints/pth"
    if feature_type.lower().startswith("aliked"):
        local_aliked_path = os.path.join(local_weights_root, "checkpoints/pth/aliked-n16rot.pth")
        extractor = ALIKED(local_path=local_aliked_path, max_num_keypoints=max_num_keypoints).eval().to(device)
    elif feature_type.lower() == "superpoint":
        local_superpoint_path = os.path.join(local_weights_root, "checkpoints/pth/superpoint_v1.pth")
        extractor = SuperPoint(path_or_url=local_superpoint_path, max_num_keypoints=max_num_keypoints).eval().to(device)
    else:
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)

    # Get list of images
    image_files = sorted(images_path)
    
    if not image_files:
        logger.error(f"No images found in {images_path}")
        return {}, {}, 0
    
    logger.info(f"Found {len(image_files)} images")
    
    # Connect to COLMAP database
    try:
        db = COLMAPDatabase.connect(database_path)
        cursor = db.cursor()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return {}, {}, 0
    
    # Extract features for all images
    logger.info("Extracting features with SuperPoint...")
    logger.info(f"Device: {device}, Processing {len(image_files)} images")
    feature_start = time.time()
    
    image_features = {}
    image_id_map = {}
    total_keypoints = 0
    
    # Buffer for batch write operations
    keypoints_to_write = []
    extraction_log = []
    
    for idx, img_path in enumerate(image_files):
        
        # Log memory status before processing each image (for GPU debugging)
        if device == 'cuda':
            try:
                mem_alloc = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory before {idx+1} Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
            except Exception as mem_e:
                logger.warning("some unknow exception !!!")
                pass
        
        try:
            # Load image
            # logger.info(f"{idx+1}/{len(image_files)} Loading image: {img_path}")
            img_tensor = load_image_use_torchvision(img_path, None, logger)
            # img_u8 = load_image_use_PIL(img_path, None, logger)
            # logger.info("to cp data to gpu")
            # img_tensor = img_u8.to(device, non_blocking=False)
            # logger.info("fininshed cp data to gpu")
            # del img_u8
            # logger.info("to cp data to gpu and convert uint8 to float")
            img_tensor = img_tensor.to(device, non_blocking=False).float().div_(255.0)
            # img_tensor = safe_load_image(image_path=img_path, logger=logger).to(device).div_(255.0)
            # img_tensor = load_image(img_path).to(device) / 255.0
            # logger.debug(f"[{idx+1}] Image shape: {img_tensor.shape}")

            if img_tensor is None:
                logger.warning(f"failed to load image: {img_path}")
                continue
            
            # Extract features
            # logger.info(f"{idx+1}/{len(image_files)} Extracting features...")
            with torch.no_grad():
                feats = extractor.extract(img_tensor)
            # logger.info(f"[{idx+1}] Feature extraction done, keypoints num: {feats['keypoints'].shape}")
            
            # Get image ID from database
            # logger.info("to fetch image id in database ")
            img_rel_path = os.path.relpath(img_path, images_dir)
            cursor.execute("SELECT image_id FROM images WHERE name = ?", (img_rel_path,))
            result = cursor.fetchone()
            
            if result is None:
                logger.warning(f"Image {img_rel_path} not found in database, skipping feature write")
                image_features[img_rel_path] = feats
                image_id_map[img_rel_path] = None
                # Clear GPU memory
                del img_tensor
                if device == 'cuda':
                    torch.cuda.empty_cache()
                continue
            
            image_id = result[0]
            image_features[img_rel_path] = feats
            image_id_map[img_rel_path] = image_id
            
            # Extract keypoints and descriptors (remove batch dimension)
            keypoints_batch = feats['keypoints']
            descriptors_batch = feats['descriptors']
            
            # Remove batch dimension if present
            if keypoints_batch.dim() == 3 and keypoints_batch.shape[0] == 1:
                keypoints = keypoints_batch[0]
                descriptors = descriptors_batch[0]
            else:
                keypoints = keypoints_batch
                descriptors = descriptors_batch
            
            # Convert to numpy
            # logger.info("convert keypoint to cpu")
            keypoints = keypoints.cpu().numpy()
            descriptors = descriptors.cpu().numpy()
            
            # Descriptors from SuperPoint are Float32 in range [0, 1], convert to uint8
            descriptors = (descriptors * 255.0).astype(np.uint8)
            
            num_kpts = keypoints.shape[0]
            total_keypoints += num_kpts
            
            # Buffer keypoints for batch write
            keypoints_to_write.append((image_id, keypoints, descriptors))
            extraction_log.append((idx, img_rel_path, num_kpts, image_id))

            logger.info(f"{idx+1}/{len(image_files)} {img_rel_path}: image_id = {image_id}, keypoints_num = {num_kpts}")
            
            # Clear GPU memory after each image
            del img_tensor, feats
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {img_path}: {e}", exc_info=True)
            image_features[img_rel_path] = None
            # Ensure cleanup on error
            try:
                del img_tensor
            except:
                pass
            if device == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            continue
    
    # Batch write all keypoints and descriptors to database (optimized - single transaction)
    logger.info(f"Writing {len(keypoints_to_write)} images' keypoints to database in batch mode...")
    feature_type_int =  {"SIFT": 1, "ALIKED_N16ROT": 2, "ALIKED_N32": 3, "SUPERPOINT": 4}.get(feature_type.upper(), 0)
    logger.info(f"Feature type: {feature_type}, feature_type_int: {feature_type_int}")
    write_start = time.time()
    written_images = batch_write_keypoints_to_database(db, keypoints_to_write, feature_type=feature_type_int, logger=logger)
    write_time = time.time() - write_start
    logger.info(f"Batch keypoints write completed in {write_time:.2f}s ({written_images} images written)")
    
    # Log extraction results
    # for idx, img_file, num_kpts, image_id in extraction_log:
    #     logger.info(f"  [{idx+1}/{len(image_files)}] {img_file}: {num_kpts} keypoints (image_id={image_id})")
    
    feature_extraction_time = time.time() - feature_start
    successful_extractions = sum(1 for f in image_features.values() if f is not None)
    logger.info(f"Feature extraction completed in {feature_extraction_time:.2f}s")
    logger.info(f"  Extracted features from {successful_extractions}/{len(image_files)} images")
    logger.info(f"  Total keypoints extracted: {total_keypoints}")
    if successful_extractions > 0:
        logger.info(f"  Average keypoints per image: {total_keypoints/successful_extractions:.1f}")
    
    db.close()
    
    return image_features, image_id_map, feature_extraction_time

