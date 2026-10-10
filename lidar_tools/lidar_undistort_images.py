#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Undistort fisheye images using camera intrinsics from a calibration.json file.

The camera intrinsics, distortion coefficients and image size are read from the
`calibration.json` file.  Every image found in the input directory is
undistorted with the intrinsics of its matching camera and written to the
output directory while preserving the original file name.

Usage:
    python undistort_images.py --calib_path <calibration.json> \
        --input_dir <image_dir> --output_dir <out_dir> [options]

Required arguments:
    --calib_path    path to calibration.json
    --input_dir     directory holding the images to undistort
    --output_dir    directory that receives the undistorted images

Layout handling:
    The script supports two input layouts, detected automatically:

    1. Multi-camera layout: the input directory contains one sub-directory per
       camera, named after the camera "name" field in calibration.json
       (e.g. ``<input_dir>/left`` and ``<input_dir>/right``).  Each camera is
       processed independently and the result is written to
       ``<output_dir>/<camera_name>/``.

    2. Single-camera layout: the input directory directly contains the images
       of one camera.  The camera is selected with ``--camera <name>``; when
       omitted it is inferred if calibration.json defines exactly one camera.
       The result is written directly to ``<output_dir>``.

    ``--camera`` can also be used to restrict a multi-camera run to a subset of
    cameras, e.g. ``--camera left``.

Common options:
    --camera NAME[,...]      camera(s) to process (default: auto-detect)
    --interp {nearest,linear,cubic,lanczos4}
                             interpolation used by cv2.remap (default: cubic)
    --max_workers INT        parallel workers; 0 = auto (60% of CPU cores)
    --keep_principal_point   keep the calibrated cx/cy instead of forcing the
                             principal point to the image center
    --recursive              also look for images in nested sub-directories
    --report                 write undistort_report.json into the output root

calibration.json is expected to look like:
    {
        "cameras": [
            {
                "name": "left",
                "width": 2912,
                "height": 2912,
                "intrinsic": {"fl_x": ..., "fl_y": ..., "cx": ..., "cy": ...},
                "distortion": {
                    "camera_model": "OPENCV_FISHEYE",
                    "params": {"k1": ..., "k2": ..., "k3": ..., "k4": ...}
                }
            }
        ]
    }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# 获取当前文件所在目录的上一级目录
sys.path.append(str(Path(__file__).parent.parent))
from unicode_paths import imread as unicode_imread, imwrite as unicode_imwrite

# Default undistortion worker ratio: use 60% of logical CPU cores.
_DEFAULT_WORKER_RATIO = 0.60

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tif", ".tiff"}

_INTERP_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}

# Models handled through cv2.fisheye (equidistant distortion model, k1..k4).
_FISHEYE_MODELS = {
    "OPENCV_FISHEYE",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
    "RAD_TAN_THIN_PRISM_FISHEYE",
}

# Models that carry no distortion coefficients.
_PINHOLE_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def load_cameras(calib_path):
    """Load every named camera from calibration.json.

    Returns:
        Dict mapping camera name -> raw camera config dict.
    """
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    cameras = {}
    for cam in calib.get("cameras", []):
        name = cam.get("name")
        if not name:
            continue
        cameras[name] = cam

    if not cameras:
        raise ValueError(f"No named cameras found in {calib_path}")
    return cameras


def make_K(intrinsic):
    """Build a 3x3 camera matrix from an intrinsic dict."""
    return np.array([
        [intrinsic["fl_x"], 0.0, intrinsic["cx"]],
        [0.0, intrinsic["fl_y"], intrinsic["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def camera_model(cam):
    """Return the upper-cased distortion model name of a camera."""
    distortion = cam.get("distortion") or {}
    return str(distortion.get("camera_model", "OPENCV_FISHEYE")).upper()


def build_distortion(cam):
    """Return the OpenCV distortion coefficient array for a camera.

    Fisheye models use ``[k1, k2, k3, k4]``.  Perspective models use
    ``[k1, k2, p1, p2, k3, k4, k5, k6]`` truncated to the coefficients that are
    actually present.  Pinhole models return an empty array.
    """
    distortion = cam.get("distortion") or {}
    params = distortion.get("params") or {}
    model = camera_model(cam)

    if model in _PINHOLE_MODELS:
        return np.zeros(0, dtype=np.float64)

    if model in _FISHEYE_MODELS:
        return np.array([
            float(params.get("k1", 0.0)),
            float(params.get("k2", 0.0)),
            float(params.get("k3", 0.0)),
            float(params.get("k4", 0.0)),
        ], dtype=np.float64)

    # Perspective models: keep the standard OpenCV ordering.
    ordered_keys = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    values = [float(params[key]) for key in ordered_keys if key in params]
    if not values:
        return np.zeros(0, dtype=np.float64)
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def list_images(image_dir, recursive=False):
    """Return the sorted list of image file paths found in ``image_dir``."""
    if recursive:
        candidates = [
            p for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
            if os.path.isfile(p)
        ]
    else:
        candidates = [
            p for p in glob.glob(os.path.join(image_dir, "*"))
            if os.path.isfile(p)
        ]
    paths = [
        p for p in candidates
        if os.path.splitext(p)[1].lower() in _IMAGE_EXTS
    ]
    paths.sort()
    return paths


# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------

def undistort_camera(name, cam, src_dir, out_dir, interp=cv2.INTER_CUBIC,
                     max_workers=4, keep_principal_point=False,
                     recursive=False):
    """Undistort every image of one camera, returning the new intrinsics.

    The remap maps are computed once per camera and reused for all images.
    The calibrated focal lengths are preserved; the principal point is moved to
    the image centre unless ``keep_principal_point`` is set.

    Returns:
        Tuple ``(intrinsic, count)`` where ``intrinsic`` is a dict with the new
        pinhole intrinsics and ``count`` is the number of written images.
    """
    os.makedirs(out_dir, exist_ok=True)

    paths = list_images(src_dir, recursive=recursive)
    if not paths:
        raise RuntimeError(f"No images found for camera {name} in {src_dir}")

    K = make_K(cam["intrinsic"])
    D = build_distortion(cam)
    model = camera_model(cam)

    width = int(cam["width"])
    height = int(cam["height"])

    target_intrinsic = cam["intrinsic"].copy()
    if not keep_principal_point:
        target_intrinsic["cx"] = width / 2.0
        target_intrinsic["cy"] = height / 2.0
    new_K = make_K(target_intrinsic)
    new_size = (width, height)

    # Warn when the real image size diverges from the calibration size.
    sample = unicode_imread(paths[0])
    if sample is None:
        raise RuntimeError(f"Cannot read sample image: {paths[0]}")
    sample_h, sample_w = sample.shape[:2]
    if (sample_w, sample_h) != new_size:
        print(f"[{name}] [warn] image size {sample_w}x{sample_h} differs from "
              f"calibration size {width}x{height}")

    if model in _PINHOLE_MODELS or D.size == 0:
        map1, map2 = cv2.initUndistortRectifyMap(
            K, None, None, new_K, new_size, cv2.CV_16SC2
        )
    elif model in _FISHEYE_MODELS:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, new_size, cv2.CV_16SC2
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, new_size, cv2.CV_16SC2
        )

    def worker(path):
        img = unicode_imread(path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        undistorted = cv2.remap(img, map1, map2, interp)
        out_path = os.path.join(out_dir, os.path.basename(path))
        if not unicode_imwrite(out_path, undistorted):
            raise RuntimeError(f"Cannot write image: {out_path}")
        return out_path

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, p) for p in paths]
        for fut in as_completed(futures):
            fut.result()

    intrinsic = {
        "fl_x": float(new_K[0, 0]),
        "fl_y": float(new_K[1, 1]),
        "cx": float(new_K[0, 2]),
        "cy": float(new_K[1, 2]),
        "width": width,
        "height": height,
    }
    print(f"[{name}] undistorted {len(paths)} images ({model}) -> {out_dir}")
    print(f"[{name}] new PINHOLE intrinsics: fx={intrinsic['fl_x']:.3f} "
          f"fy={intrinsic['fl_y']:.3f} cx={intrinsic['cx']:.3f} "
          f"cy={intrinsic['cy']:.3f} ({width}x{height})")
    return intrinsic, len(paths)


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------

def build_plan(cameras, input_dir, output_dir, camera_arg):
    """Return the list of ``(name, cam, src_dir, out_dir)`` work items.

    ``camera_arg`` is the parsed ``--camera`` value (a list or None).
    """
    subdirs = {
        entry for entry in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, entry))
    }
    matching = sorted(n for n in cameras if n in subdirs)

    # Case 1: an explicit subset of cameras was requested.
    if camera_arg:
        plan = []
        for name in camera_arg:
            if name not in cameras:
                raise ValueError(
                    f"Camera {name!r} not found in calibration.json "
                    f"(available: {sorted(cameras)})"
                )
            cam_subdir = os.path.join(input_dir, name)
            if os.path.isdir(cam_subdir):
                src = cam_subdir
                dst = os.path.join(output_dir, name)
            else:
                if len(camera_arg) > 1:
                    raise ValueError(
                        f"Cannot map camera {name!r}: no sub-directory "
                        f"{cam_subdir} exists for multi-camera input"
                    )
                src = input_dir
                dst = output_dir
            plan.append((name, cameras[name], src, dst))
        return plan

    # Case 2: auto-detect a multi-camera layout.
    if matching:
        return [
            (name, cameras[name],
             os.path.join(input_dir, name),
             os.path.join(output_dir, name))
            for name in matching
        ]

    # Case 3: single-camera layout.
    if len(cameras) == 1:
        name = next(iter(cameras))
        return [(name, cameras[name], input_dir, output_dir)]

    raise ValueError(
        "Could not determine the camera for the input directory. It contains "
        f"no sub-directory matching a calibrated camera ({sorted(cameras)}). "
        "Pass --camera <name> to select one."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Undistort images using intrinsics from calibration.json"
    )
    parser.add_argument("--calib_path", type=str, required=True,
                        help="path to calibration.json")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="directory containing the images to undistort")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="directory to write undistorted images into")
    parser.add_argument("--camera", type=str, default=None,
                        help="comma-separated camera name(s) to process; "
                             "default auto-detects the layout")
    parser.add_argument("--interp", choices=sorted(_INTERP_MAP), default="cubic",
                        help="interpolation used by cv2.remap (default: cubic)")
    parser.add_argument("--max_workers", type=int, default=0,
                        help="parallel workers; 0 means auto (60%% of CPU cores)")
    parser.add_argument("--keep_principal_point", action="store_true",
                        help="keep the calibrated cx/cy instead of forcing the "
                             "principal point to the image center")
    parser.add_argument("--recursive", action="store_true",
                        help="also look for images in nested sub-directories")
    parser.add_argument("--report", action="store_true",
                        help="write undistort_report.json into the output root")
    args = parser.parse_args()

    if not os.path.isfile(args.calib_path):
        raise FileNotFoundError(f"calibration.json not found: {args.calib_path}")
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"input directory not found: {args.input_dir}")

    cameras = load_cameras(args.calib_path)
    print(f"[calib] loaded cameras: {sorted(cameras)}")

    camera_arg = None
    if args.camera:
        camera_arg = [c.strip() for c in args.camera.split(",") if c.strip()]

    plan = build_plan(cameras, args.input_dir, args.output_dir, camera_arg)

    interp = _INTERP_MAP[args.interp]
    if args.max_workers > 0:
        max_workers = args.max_workers
    else:
        max_workers = max(1, int((os.cpu_count() or 1) * _DEFAULT_WORKER_RATIO))
    print(f"[undistort] interpolation={args.interp}, max_workers={max_workers}")

    report = {}
    total_images = 0
    for name, cam, src_dir, out_dir in plan:
        print(f"[{name}] source: {src_dir}")
        intrinsic, count = undistort_camera(
            name, cam, src_dir, out_dir,
            interp=interp,
            max_workers=max_workers,
            keep_principal_point=args.keep_principal_point,
            recursive=args.recursive,
        )
        report[name] = {
            "source_dir": os.path.abspath(src_dir),
            "output_dir": os.path.abspath(out_dir),
            "num_images": count,
            "camera_model": camera_model(cam),
            "intrinsic": intrinsic,
        }
        total_images += count

    if args.report:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, "undistort_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {report_path}")

    print(f"[done] undistorted {total_images} image(s) "
          f"for {len(plan)} camera(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
