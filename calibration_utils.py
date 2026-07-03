"""
Calibration utilities for parsing camera calibration JSON files.

Provides functions to load per-camera intrinsics and distortion parameters
from a calibration.json file, and construct COLMAP-compatible camera parameter
lists for use in SfM pipelines.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional


def load_calibration(
    calib_path: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse a calibration.json file and return per-camera configurations
    suitable for COLMAP database initialization.

    The JSON is expected to have the structure:
        {
            "cameras": [
                {
                    "name": "left",
                    "width": 2912,
                    "height": 2912,
                    "intrinsic": {"fl_x": ..., "fl_y": ..., "cx": ..., "cy": ...},
                    "distortion": {
                        "camera_model": "OPENCV_FISHEYE",
                        "params": {"k1": ..., "k2": ..., ...}
                    }
                },
                ...
            ]
        }

    Args:
        calib_path: Path to calibration.json file.
        logger: Optional logger instance.

    Returns:
        Dict mapping camera name to config dict:
            {
                "camera_model": str,     # COLMAP camera model name
                "params": List[float],   # Full param list: [fx, fy, cx, cy, ...dist...]
                "width": int,
                "height": int,
            }
        Returns an empty dict if the file cannot be read or has no cameras.
    """
    if logger is None:
        logger = logging.getLogger()

    if not os.path.isfile(calib_path):
        logger.warning(f"Calibration file not found: {calib_path}")
        return {}

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read calibration file {calib_path}: {e}")
        return {}

    cameras = calib_data.get("cameras", [])
    if not cameras:
        logger.warning(f"No cameras found in calibration file: {calib_path}")
        return {}

    # Map COLMAP model names to expected distortion param count (excluding fx,fy,cx,cy)
    _MODEL_DISTORTION_COUNTS = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 0,
        "SIMPLE_RADIAL": 1,
        "RADIAL": 2,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 4,
        "FULL_OPENCV": 8,
        "FOV": 1,
        "SIMPLE_RADIAL_FISHEYE": 1,
        "RADIAL_FISHEYE": 2,
        "THIN_PRISM_FISHEYE": 8,
        "RAD_TAN_THIN_PRISM_FISHEYE": 12,
    }

    # --- Process each camera entry ---
    configs: Dict[str, Dict[str, Any]] = {}

    for cam in cameras:
        name = cam.get("name", "")
        if not name:
            logger.warning("Skipping camera entry with no name.")
            continue

        intrinsic = cam.get("intrinsic", {})
        fl_x = intrinsic.get("fl_x", 0.0)
        fl_y = intrinsic.get("fl_y", 0.0)
        cx = intrinsic.get("cx", 0.0)
        cy = intrinsic.get("cy", 0.0)

        distortion = cam.get("distortion", {})
        camera_model = distortion.get("camera_model", "PINHOLE").upper()
        dist_params_dict = distortion.get("params", {})

        # Remap camera model based on raw distortion params (before flattening)
        # e.g. "PINHOLE" + {k1,k2,p1,p2} → "OPENCV"
        camera_model = _remap_camera_model(
            camera_model, dist_params_dict, logger, camera_name=name
        )

        # Build distortion params in canonical order for the resolved model
        dist_params_ordered = _flatten_distortion_params(dist_params_dict, camera_model)

        expected_dist_count = _MODEL_DISTORTION_COUNTS.get(camera_model)
        if expected_dist_count is not None and len(dist_params_ordered) != expected_dist_count:
            logger.warning(
                f"Camera '{name}': model '{camera_model}' expects "
                f"{expected_dist_count} distortion params, but got "
                f"{len(dist_params_ordered)}. Using what's provided."
            )

        # Full COLMAP params: fx, fy, cx, cy, ...distortion...
        full_params = [fl_x, fl_y, cx, cy] + dist_params_ordered

        configs[name] = {
            "camera_model": camera_model,
            "params": full_params,
            "width": cam.get("width", 0),
            "height": cam.get("height", 0),
        }

        logger.info(
            f"Loaded calibration for camera '{name}': "
            f"model={camera_model}, {cam.get('width')}x{cam.get('height')}, "
            f"{len(full_params)} params"
        )

    return configs

def _remap_camera_model(
    model: str,
    dist_params_dict: Dict[str, float],
    logger: logging.Logger,
    camera_name: str = "",
) -> str:
    """
    Remap calibration JSON camera model names to COLMAP-compatible names
    based on the actual distortion parameter count.

    Common calibration tools use generic names like "pinhole" or "fisheye"
    that don't correspond 1:1 to COLMAP model names when distortion is present.
    This function resolves the correct COLMAP model.
    """
    model_upper = model.upper()
    num_dist = len(dist_params_dict)

    if num_dist == 0:
        return model_upper

    _REMAP = {
        # (model, num_dist_params) → COLMAP model
        ("PINHOLE", 1): "SIMPLE_RADIAL",
        ("PINHOLE", 2): "RADIAL",
        ("PINHOLE", 4): "OPENCV",       # k1, k2, p1, p2 → OPENCV
        ("SIMPLE_PINHOLE", 1): "SIMPLE_RADIAL",
    }

    key = (model_upper, num_dist)
    if key in _REMAP:
        new_model = _REMAP[key]
        logger.info(
            f"Camera '{camera_name}': remapped model '{model_upper}' → '{new_model}' "
            f"({num_dist} distortion params)"
        )
        return new_model

    return model_upper


def _flatten_distortion_params(
    params_dict: Dict[str, float], model: str
) -> List[float]:
    """
    Flatten named distortion params into a canonical ordered list
    matching COLMAP's parameter ordering for the given camera model.

    Args:
        params_dict: Dict of param name → value (e.g., {"k1": 0.1, "k2": -0.05}).
        model: COLMAP camera model name (uppercase).

    Returns:
        Ordered list of float distortion parameters.
    """
    # COLMAP parameter ordering per model (excluding fx,fy,cx,cy):
    _MODEL_PARAM_ORDER: Dict[str, List[str]] = {
        "SIMPLE_PINHOLE": [],
        "PINHOLE": [],
        "SIMPLE_RADIAL": ["k1"],
        "RADIAL": ["k1", "k2"],
        "OPENCV": ["k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["k1", "k2", "k3", "k4"],
        "FULL_OPENCV": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "FOV": ["omega"],
        "SIMPLE_RADIAL_FISHEYE": ["k1"],
        "RADIAL_FISHEYE": ["k1", "k2"],
        "THIN_PRISM_FISHEYE": ["k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"],
        "RAD_TAN_THIN_PRISM_FISHEYE": [
            "k0", "k1", "k2", "k3", "k4", "k5",
            "p0", "p1",
            "s0", "s1", "s2", "s3",
        ],
    }

    param_order = _MODEL_PARAM_ORDER.get(model, [])
    result = []
    for key in param_order:
        # Also try alternate naming: "omega" might be in params as "omega"
        val = params_dict.get(key)
        if val is not None:
            result.append(float(val))
    return result


def map_subfolders_to_camera_configs(
    calibration_configs: Dict[str, Dict[str, Any]],
    subfolder_names: List[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Map image subfolder names to their corresponding camera configurations
    from calibration data.

    Matching strategy:
    1. First try exact match between subfolder name and camera name.
    2. If no exact match, try case-insensitive match.
    3. If still no match, log a warning and skip.

    Args:
        calibration_configs: Output from load_calibration().
        subfolder_names: List of subfolder basenames under images_dir.
        logger: Optional logger instance.

    Returns:
        Dict mapping subfolder_name → camera config dict (same format as
        calibration_configs values).
    """
    if logger is None:
        logger = logging.getLogger()

    result = {}
    # Build lowercase lookup for case-insensitive fallback
    calib_lower = {k.lower(): k for k in calibration_configs}

    for subfolder in subfolder_names:
        basename = os.path.basename(subfolder)
        if basename in calibration_configs:
            result[basename] = calibration_configs[basename]
        elif basename.lower() in calib_lower:
            matched_name = calib_lower[basename.lower()]
            result[basename] = calibration_configs[matched_name]
            logger.info(
                f"Mapped subfolder '{basename}' → camera '{matched_name}' "
                f"(case-insensitive)"
            )
        else:
            logger.warning(
                f"Subfolder '{basename}' has no matching camera in calibration. "
                f"Available cameras: {list(calibration_configs.keys())}"
            )

    return result
