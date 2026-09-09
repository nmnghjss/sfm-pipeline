#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a COLMAP model from real-time odom + stereo fisheye images + lidar point cloud.

Usage:
    python real_time_to_colmap.py --data_dir <DATA_DIR> --output_dir <OUTPUT_DIR> [options]

Required arguments:
    --data_dir      input data directory (see expected layout below)
    --output_dir    output COLMAP directory (will be created if not exist)

Common options:
    --fmt {txt,bin,both}     sparse model output format (default: bin)
                             use 'bin' for the pipeline and COLMAP GUI
                             use 'txt' only when human-readable output is needed
                             use 'both' if unsure
    --align_mode {mean,start,end,none}  how to align image and odom timestamps
                                        default: none (images are already in odom clock)
    --num_points INT         target number of points in two-stage down-sampling
                             default: 50000
    --max_workers INT        parallel workers for undistortion; 0 = auto (60% of CPU cores)
    --undistort_interp {nearest,linear,cubic,lanczos4}
                             interpolation used by cv2.remap during undistortion
                             (default: lanczos4 for best quality)
    --skip_extract           skip extracting images from data_raw.mcap
                             (assume <output_dir>/cameras/left and right already exist)
    --skip_undistort         skip undistortion, assume images/{left,right}/
                             already exist in output_dir
    --skip_pointcloud        skip point cloud processing

Examples:
    # Basic usage: extract from mcap + fixed 1600x1600 undistortion + txt output
    python real_time_to_colmap.py --data_dir G:/Data/Laser_data/2026-06-22_15-04-26rrrr --output_dir G:/Data/Laser_data/colmap_output

    # Reuse already-extracted/undistorted images, skip heavy steps
    python real_time_to_colmap.py --data_dir ... --output_dir ... --skip_extract --skip_undistort --skip_pointcloud

Inputs (under data_dir):
    data/data_raw.mcap            raw mcap containing /camera/left/jpeg and /camera/right/jpeg
    info/calibration.json         camera intrinsics, distortion, T_lidar_to_camera
    odom-realtime.csv             IMU/odom trajectory (T_world_to_body)
    colorized-realtime.las        colored lidar point cloud

Outputs (under output_dir):
    cameras/left/*.jpg            extracted left fisheye images, filename = nanosec timestamp
    cameras/right/*.jpg           extracted right fisheye images, filename = nanosec timestamp
    images/left/*.jpg             undistorted left images
    images/right/*.jpg            undistorted right images
    sparse/0/cameras.txt          two PINHOLE cameras (camera 1 = left, 2 = right)
    sparse/0/images.txt           image poses (T_camera_to_world for COLMAP)
    sparse/0/points3D.txt         down-sampled colored point cloud with normals

Notes:
    - The script now extracts images from data_raw.mcap using the mcap's local
      publish_time, so the extracted filenames are already in the same clock as
      odom-realtime.csv. Images whose timestamp falls outside the odom range
      (e.g. the first 2 left/right frames that precede the first odom pose)
      are automatically dropped.
    - The script puts left/right images into separate subdirectories. COLMAP
      image names are stored as "left/xxx.jpg" / "right/xxx.jpg" relative to the
      images/ folder.
"""

import os
import time
import argparse
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import laspy
from scipy.spatial.transform import Slerp, Rotation as R
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from unicode_paths import imread as unicode_imread, imwrite as unicode_imwrite

# Default undistortion worker ratio: use 60% of logical CPU cores.
_DEFAULT_WORKER_RATIO = 0.60

from read_write_model import (
    Camera,
    Image,
    Point3D,
    write_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_calibration(calib_path):
    """Load calibration.json and return left/right camera dicts."""
    with open(calib_path, "r") as f:
        calib = json.load(f)

    cams = {}
    for cam in calib["cameras"]:
        name = cam["name"]
        if name not in ("left", "right"):
            continue
        cams[name] = cam
    return cams, calib.get("imu", [])


def parse_image_stamps(camera_dir):
    """Return sorted image timestamps (ns) and paths for supported formats."""
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    paths = [
        p for p in glob.glob(os.path.join(camera_dir, "*"))
        if os.path.splitext(p)[1].lower() in image_exts
    ]
    stamps, names = [], []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        stamps.append(int(base))
        names.append(os.path.basename(p))
    idx = np.argsort(stamps)
    return np.array(stamps, dtype=np.float64)[idx], np.array(paths)[idx], np.array(names)[idx]


def load_odom(csv_path):
    """Load odom CSV: timestamp x y z qx qy qz qw ..."""
    df = pd.read_csv(csv_path, comment="#", header=None)
    stamps = np.ascontiguousarray(df.iloc[:, 0].to_numpy(dtype=np.float64))
    xyz = np.ascontiguousarray(df.iloc[:, 1:4].to_numpy(dtype=np.float64))
    quat = np.ascontiguousarray(df.iloc[:, 4:8].to_numpy(dtype=np.float64))  # qx qy qz qw
    return stamps, xyz, quat


def compute_time_offset(img_stamps, odom_stamps, mode="none"):
    """
    Both sequences may have the same duration but a constant clock offset.
    mode: 'mean' | 'start' | 'end' | 'none'
    """
    if mode == "mean":
        return np.mean(img_stamps) - np.mean(odom_stamps)
    elif mode == "start":
        return np.min(img_stamps) - np.min(odom_stamps)
    elif mode == "end":
        return np.max(img_stamps) - np.max(odom_stamps)
    elif mode == "none":
        return 0.0
    else:
        raise ValueError(f"Unknown align mode: {mode}")


def make_K(intrinsic):
    return np.array([
        [intrinsic["fl_x"], 0.0, intrinsic["cx"]],
        [0.0, intrinsic["fl_y"], intrinsic["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Point-cloud sampling
# ---------------------------------------------------------------------------

def voxel_random_sample(point_cloud, voxel_size, target_num_points,
                        rng=None):
    """Randomly sample points from every voxel.

    The per-voxel sample count is ``ceil(target_num_points / voxel_count)``.
    If a voxel contains fewer points than this count, all of its points are
    retained.  The result can therefore contain slightly more than the target
    count, as required by the per-voxel sampling rule.

    Args:
        point_cloud: ``(N, D)`` array. The first three columns are XYZ;
            additional columns, such as RGB, are kept with each point.
        voxel_size: Positive voxel edge length in the same units as XYZ.
        target_num_points: Desired total sampling budget.
        rng: Optional ``numpy.random.Generator`` for reproducible sampling.

    Returns:
        A sampled ``(M, D)`` array. The input is not modified.
    """
    points = np.asarray(point_cloud)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point_cloud must have shape (N, D) with D >= 3")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if target_num_points < 0:
        raise ValueError("target_num_points must be non-negative")
    if len(points) == 0 or target_num_points == 0:
        return points[:0].copy()

    rng = np.random.default_rng() if rng is None else rng
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    points_per_voxel = int(np.ceil(target_num_points / voxel_count))
    print(f"[voxel_random_sample] voxel_size={voxel_size}, voxel_count={voxel_count}, points_per_voxel={points_per_voxel}")

    # Sort once by voxel and a random key. The first points in each voxel
    # are therefore an unbiased sample without repeatedly scanning inverse.
    random_keys = rng.random(len(inverse))
    order = np.lexsort((random_keys, inverse))
    sorted_inverse = inverse[order]

    voxel_starts = np.r_[0, np.flatnonzero(np.diff(sorted_inverse)) + 1]
    voxel_ends = np.r_[voxel_starts[1:], len(order)]
    voxel_counts = voxel_ends - voxel_starts
    sample_counts = np.minimum(voxel_counts, points_per_voxel)

    group_offsets = np.arange(len(order)) - np.repeat(voxel_starts, voxel_counts)
    selected_mask = group_offsets < np.repeat(sample_counts, voxel_counts)
    selected = order[selected_mask]

    selected.sort()
    result = points[selected].copy()
    return result


def two_stage_point_cloud_sample(point_cloud, target_num_points, voxel_size,
                                 random_ratio = 0.5, rng=None):
    """Down-sample a point cloud using random and voxel sampling stages.

    First, randomly samples half of ``target_num_points`` from the input.
    The selected points are removed.  The second half is sampled from the
    remaining points using :func:`voxel_random_sample`.

    If the input contains fewer points than the target, all points are
    returned.  For odd targets, the first stage receives the smaller half and
    the second stage receives the remaining budget.

    Args:
        point_cloud: ``(N, D)`` array with XYZ in the first three columns.
        target_num_points: Maximum number of points to return.
        voxel_size: Positive voxel edge length.
        rng: Optional ``numpy.random.Generator`` for reproducibility.

    Returns:
        A sampled ``(M, D)`` array with ``M <= target_num_points``.
    """
    points = np.asarray(point_cloud)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point_cloud must have shape (N, D) with D >= 3")
    if target_num_points < 0:
        raise ValueError("target_num_points must be non-negative")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(points) == 0 or target_num_points == 0:
        return points[:0].copy()
    if len(points) <= target_num_points:
        return points.copy()

    rng = np.random.default_rng() if rng is None else rng
    first_stage_target = int(target_num_points * random_ratio)
    second_stage_target = target_num_points - first_stage_target

    first_indices = rng.choice(
        len(points), first_stage_target, replace=False
    )

    selected_mask = np.ones(len(points), dtype=bool)
    selected_mask[first_indices] = False
    remaining = points[selected_mask]

    second_stage = voxel_random_sample(
        remaining,
        voxel_size=voxel_size,
        target_num_points=second_stage_target,
        rng=rng,
    )
    result = np.concatenate([points[first_indices], second_stage], axis=0)

    # The per-voxel rule can exceed its requested budget. Enforce the public
    # total-budget contract while retaining both sampling stages.
    if len(result) > target_num_points:
        result = result[rng.choice(len(result), target_num_points, replace=False)]
    return result


# ---------------------------------------------------------------------------
# MCAP image extraction
# ---------------------------------------------------------------------------

def _image_ext_from_format(fmt):
    fmt_lower = fmt.lower().split(";", 1)[0].strip()
    if "/" in fmt_lower:
        fmt_lower = fmt_lower.rsplit("/", 1)[-1]
    if fmt_lower in ("jpeg", "jpg"):
        return "jpg"
    if fmt_lower == "png":
        return "png"
    if fmt_lower == "webp":
        return "webp"
    if fmt_lower == "avif":
        return "avif"
    return "bin"


def _payload_signature(data):
    """Return a useful signature for common compressed image payloads."""
    payload = bytes(data)
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "webp"
    if payload.startswith(b"\x00\x00\x00") and b"ftypavif" in payload[:32]:
        return "avif"
    return f"unknown ({payload[:12].hex()})"


def _compressed_image_type_name(schema_name):
    """Normalize ROS1/ROS2 CompressedImage names for matching and decoding."""
    normalized = schema_name.replace("/msg/", "/")
    if normalized in ("foxglove_msgs/CompressedImage", "sensor_msgs/CompressedImage"):
        return schema_name
    return None


def _ros_type_key(schema_name):
    """Return the rosbags type key for ROS1 or ROS2 schema names."""
    if "/msg/" in schema_name:
        return schema_name
    return schema_name.replace("/", "/msg/", 1)


def _deserialize_compressed_image(typestore, message_data, msg_type_key,
                                  message_encoding):
    """Deserialize a ROS CompressedImage using the MCAP channel encoding."""
    encoding = (message_encoding or "").lower().strip()
    if encoding in {"ros1", "ros1msg", "rosmsg"}:
        return typestore.deserialize_ros1(message_data, msg_type_key)
    if encoding in {"cdr", "ros2", "ros2msg"}:
        return typestore.deserialize_cdr(message_data, msg_type_key)
    raise RuntimeError(
        f"Unsupported MCAP message encoding {message_encoding!r}; "
        "expected ros1/ros1msg or cdr/ros2msg"
    )


def extract_images_from_mcap(mcap_path, cameras_dir, time_field="publish_time"):
    """
    Extract /camera/left/jpeg and /camera/right/jpeg from an mcap file.
    Output layout: <cameras_dir>/<left|right>/<timestamp>.<format>.
    Returns a dict {topic: count}.
    """
    os.makedirs(cameras_dir, exist_ok=True)
    topic_to_dir = {
        "/camera/left/jpeg": "left",
        "/camera/right/jpeg": "right",
    }

    counts = {}
    with open(mcap_path, "rb") as f:
        # Detect corrupted records/chunks instead of silently writing damaged
        # compressed payloads to image files.
        reader = make_reader(f, validate_crcs=True)
        summary = reader.get_summary()

        # Register and locate compressed image channels.  The channel encoding
        # determines whether the serialized ROS message is ROS1 or CDR/ROS2.
        image_channels = {}
        for cid, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            if schema is None:
                continue
            if _compressed_image_type_name(schema.name) is None:
                continue
            if channel.topic not in topic_to_dir:
                continue

            message_encoding = (channel.message_encoding or "").lower().strip()
            if message_encoding in {"cdr", "ros2", "ros2msg"}:
                typestore = get_typestore(Stores.ROS2_HUMBLE)
                deserialize_kind = "CDR/ROS2"
            elif message_encoding in {"ros1", "ros1msg", "rosmsg"}:
                typestore = get_typestore(Stores.ROS1_NOETIC)
                deserialize_kind = "ROS1"
            else:
                raise RuntimeError(
                    f"Unsupported encoding for {channel.topic}: "
                    f"message_encoding={channel.message_encoding!r}, "
                    f"schema_encoding={schema.encoding!r}"
                )

            typestore.register(get_types_from_msg(schema.data.decode("utf-8"), schema.name))
            msg_type_key = _ros_type_key(schema.name)
            out_dir = os.path.join(cameras_dir, topic_to_dir[channel.topic])
            os.makedirs(out_dir, exist_ok=True)
            image_channels[cid] = {
                "topic": channel.topic,
                "msg_type_key": msg_type_key,
                "message_encoding": channel.message_encoding,
                "schema_encoding": schema.encoding,
                "deserialize_kind": deserialize_kind,
                "out_dir": out_dir,
                "count": 0,
                "formats": set(),
                "signatures": set(),
                "decode_failures": 0,
            }

        if not image_channels:
            raise RuntimeError(
                f"No /camera/left/jpeg or /camera/right/jpeg channels found in {mcap_path}"
            )

        print(f"[extract] found {len(image_channels)} image channel(s):")
        for info in image_channels.values():
            print(f"  {info['topic']} -> {info['out_dir']} | "
                  f"schema={info['msg_type_key']} "
                  f"schema_encoding={info['schema_encoding']!r} "
                  f"message_encoding={info['message_encoding']!r} "
                  f"decoder={info['deserialize_kind']}")

        for schema, channel, message in reader.iter_messages():
            if channel.id not in image_channels:
                continue
            info = image_channels[channel.id]

            ts = getattr(message, time_field)
            # Do not use the MCAP record timestamp as image content metadata;
            # this timestamp only controls the output filename.
            typestore = get_typestore(
                Stores.ROS2_HUMBLE
                if info["deserialize_kind"] == "CDR/ROS2"
                else Stores.ROS1_NOETIC
            )
            typestore.register(get_types_from_msg(
                schema.data.decode("utf-8"), schema.name
            ))
            img = _deserialize_compressed_image(
                typestore, message.data, info["msg_type_key"],
                info["message_encoding"]
            )
            payload = bytes(img.data)
            fmt = str(img.format)
            signature = _payload_signature(payload)
            info["formats"].add(fmt)
            info["signatures"].add(signature)
            if _image_ext_from_format(fmt) == "bin":
                raise RuntimeError(
                    f"Unsupported compressed image format {fmt!r} on {info['topic']}"
                )

            if info["count"] == 0:
                print(f"  [{info['topic']}] payload format={fmt!r}, "
                      f"signature={signature}, bytes={len(payload)}")
            ext = _image_ext_from_format(img.format)
            filename = f"{ts}.{ext}"
            out_path = os.path.join(info["out_dir"], filename)

            with open(out_path, "wb") as outf:
                outf.write(payload)

            with open(out_path, "rb") as inf:
                if inf.read() != payload:
                    raise IOError(f"Written image bytes differ from MCAP payload: {out_path}")

            info["count"] += 1
            if info["count"] % 50 == 0:
                print(f"  [{info['topic']}] extracted {info['count']} images...")

        for info in image_channels.values():
            counts[info["topic"]] = info["count"]
            print(f"  [{info['topic']}] formats={sorted(info['formats'])}, "
                  f"payload_signatures={sorted(info['signatures'])}, "
                  f"OpenCV_decode_failures={info['decode_failures']}")

    print("[extract] done:")
    for topic, count in counts.items():
        print(f"  {topic}: {count} images")
    return counts


# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------

def undistort_camera(name, cam, src_dir, out_dir,
                     target_size, target_K, max_workers=4,
                     interp=cv2.INTER_LINEAR,
                     stamps=None, paths=None, names=None):
    """
    Undistort selected images of one camera, return new PINHOLE intrinsics.

    The undistortion maps are pre-computed once per camera and reused for all
    images. The interpolation method can be chosen via ``interp``; LINEAR is
    much faster than LANCZOS4 and is usually sufficient for SfM/3DGS.
    """
    os.makedirs(out_dir, exist_ok=True)
    if stamps is None or paths is None or names is None:
        stamps, paths, names = parse_image_stamps(src_dir)

    if len(paths) == 0:
        raise RuntimeError(f"No images to undistort for camera {name}")

    # Read one sample image to determine input size and precompute maps.
    sample = unicode_imread(paths[0])
    if sample is None:
        raise RuntimeError(f"Cannot read sample image: {paths[0]}")
    K = make_K(cam["intrinsic"])
    D = np.array([
        cam["distortion"]["params"]["k1"],
        cam["distortion"]["params"]["k2"],
        cam["distortion"]["params"]["k3"],
        cam["distortion"]["params"]["k4"],
    ], dtype=np.float64)

    # Use calibration.json intrinsics and image size as the output target.
    new_K = target_K.copy()
    new_size = target_size

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, new_size, cv2.CV_16SC2
    )
    W, H = new_size

    def worker(args):
        p, out_name = args
        img = unicode_imread(p)
        if img is None:
            raise RuntimeError(f"Cannot read image: {p}")
        undistorted = cv2.remap(img, map1, map2, interp)
        out_path = os.path.join(out_dir, out_name)
        if not unicode_imwrite(out_path, undistorted):
            raise RuntimeError(f"Cannot write image: {out_path}")
        return out_name

    tasks = list(zip(paths, names))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            fut.result()

    intrinsic = {
        "fl_x": float(new_K[0, 0]),
        "fl_y": float(new_K[1, 1]),
        "cx": float(new_K[0, 2]),
        "cy": float(new_K[1, 2]),
        "width": W,
        "height": H,
    }
    print(f"[{name}] undistorted {len(paths)} images -> {out_dir}")
    print(f"[{name}] new PINHOLE intrinsics: fx={intrinsic['fl_x']:.3f} fy={intrinsic['fl_y']:.3f} "
          f"cx={intrinsic['cx']:.3f} cy={intrinsic['cy']:.3f} ({W}x{H})")
    return intrinsic, stamps, names


def interpolate_pose(t_query, ts, xyz, quat):
    """
    Interpolate world->body pose at t_query (ns).
    quat is (qx, qy, qz, qw); output rotation is scipy Rotation (scalar-last).
    """
    ts = np.ascontiguousarray(ts)
    xyz = np.ascontiguousarray(xyz)
    quat = np.ascontiguousarray(quat)

    if t_query < ts[0] or t_query > ts[-1]:
        return None, None

    # Linear position
    x = np.interp(t_query, ts, xyz[:, 0])
    y = np.interp(t_query, ts, xyz[:, 1])
    z = np.interp(t_query, ts, xyz[:, 2])
    t = np.array([x, y, z], dtype=np.float64)

    # Spherical quaternion interpolation
    rots = R.from_quat(quat)  # scipy uses (x, y, z, w)
    slerp = Slerp(ts, rots)
    r = slerp([t_query])[0]
    return r.as_matrix(), t


def invert_pose(R_mat, t_vec):
    """Invert T = [R|t]."""
    R_inv = R_mat.T
    t_inv = -R_inv @ t_vec
    return R_inv, t_inv


def pose_to_colmap_qt(R_cw, t_cw):
    """Return COLMAP qvec (qw,qx,qy,qz) and tvec."""
    q = R.from_matrix(R_cw).as_quat()  # x,y,z,w
    qvec = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return qvec, t_cw


def build_body_to_lidar(imu_calib):
    """
    T_lidar_to_imu from calibration. We need T_imu_to_lidar = inverse.
    If absent, assume identity.
    """
    if not imu_calib:
        return np.eye(3), np.zeros(3)
    imu = imu_calib[0]
    T_li = imu.get("lidar_to_imu_transform", {})
    R_li = np.array(T_li.get("rotation", np.eye(3)), dtype=np.float64)
    t_li = np.array(T_li.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    R_il, t_il = invert_pose(R_li, t_li)
    return R_il, t_il


def align_to_colmap_axis(R_wc, t_wc, mode):
    """
    设备/odom 坐标系与 COLMAP 相机坐标系之间的轴向对齐。

    COLMAP 相机坐标系：X 右、Y 下、Z 前（看向场景内部）。
    某些设备（如 AR/SLAM）输出相机前向为 -Z，需要 180° 绕 X 轴翻转。
    该函数在 (R_wc, t_wc) 是 camera-to-world 的约定下，对相机坐标系施加 R_fix。

    Parameters
    ----------
    R_wc, t_wc : camera-to-world rotation/translation.
    mode : str, one of {"none", "x180", "y180", "z180"}

    Returns
    -------
    R_wc', t_wc'
    """
    if mode == "none":
        return R_wc, t_wc
    if mode == "x180":
        R_fix = np.diag([1.0, -1.0, -1.0])
    elif mode == "y180":
        R_fix = np.diag([-1.0, 1.0, -1.0])
    elif mode == "z180":
        R_fix = np.diag([-1.0, -1.0, 1.0])
    else:
        raise ValueError(f"Unsupported axis_align mode: {mode}")
    # Right-multiply so that the final world-to-camera becomes R_fix @ R_cw,
    # matching the left-multiplication in convert_ar_to_colmap.py.
    R_wc = R_wc @ R_fix
    return R_wc, t_wc


def build_camera_pose(R_wi, t_wi, R_il, t_il, R_lc, t_lc):
    """
    T_wc = T_wi * T_il * T_lc
    """
    R_wl = R_wi @ R_il
    t_wl = R_wi @ t_il + t_wi
    R_wc = R_wl @ R_lc
    t_wc = R_wl @ t_lc + t_wl
    return R_wc, t_wc


def process_point_cloud(las_path, out_ply, voxel_size=1.0, random_ratio=0.6,
                        target_num_points=50000):
    """Load, sample, normal-estimate, and write a colored LAS point cloud.

    Sampling uses ``two_stage_point_cloud_sample``: half of the target points
    are selected uniformly at random, and the other half are selected from the
    remaining points using per-voxel random sampling.
    """
    print(f"[point cloud] reading {las_path}")
    las = laspy.read(las_path)
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    # Read colors if present
    has_color = False
    colors = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        # las color is often uint16
        r = np.array(las.red, dtype=np.float64)
        g = np.array(las.green, dtype=np.float64)
        b = np.array(las.blue, dtype=np.float64)
        if r.max() > 255:
            r /= 65535.0
            g /= 65535.0
            b /= 65535.0
        else:
            r /= 255.0
            g /= 255.0
            b /= 255.0
        colors = np.vstack([r, g, b]).T.astype(np.float64)
        has_color = True

    n_total = len(pts)
    print(f"[point cloud] total points: {n_total}, "
          f"target down-sample: {target_num_points}, voxel_size: {voxel_size}")
    if n_total > target_num_points:
        rng = np.random.default_rng(42)
        if has_color:
            point_data = np.concatenate([pts, colors], axis=1)
            sampled = two_stage_point_cloud_sample(
                point_data,
                target_num_points=target_num_points,
                voxel_size=voxel_size,
                random_ratio=random_ratio,
                rng=rng,
            )
            pts = sampled[:, :3]
            colors = sampled[:, 3:]
        else:
            pts = two_stage_point_cloud_sample(
                pts,
                target_num_points=target_num_points,
                voxel_size=voxel_size,
                random_ratio=random_ratio,
                rng=rng,
            )
        print(f"[point cloud] two-stage downsample: {n_total} -> {len(pts)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if has_color:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    print(f"[point cloud] estimating normals on {len(pcd.points)} points")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)
    print(f"[point cloud] wrote {out_ply}")
    return pcd


def build_points3D_dict(pcd):
    """Build COLMAP points3D dict from open3d PointCloud (with colors)."""
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

    points3D = {}
    for i in range(len(xyz)):
        points3D[i + 1] = Point3D(
            id=i + 1,
            xyz=xyz[i],
            rgb=rgb[i],
            error=0.0,
            image_ids=np.zeros(0, dtype=np.int32),
            point2D_idxs=np.zeros(0, dtype=np.int32),
        )
    return points3D


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class StepTimer:
    """Print step start/finish messages and produce a final timing summary."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.step_start = self.t0
        self.records = []
        self.current_label = None

    def start(self, label):
        self.current_label = label
        print(f"\n[step] >>> {label} ...")
        self.step_start = time.perf_counter()

    def finish(self):
        elapsed = time.perf_counter() - self.step_start
        label = self.current_label or "unknown"
        self.records.append((label, elapsed))
        print(f"[time] {label}: {elapsed:.2f} s")
        self.current_label = None
        return elapsed

    def summary(self):
        total = time.perf_counter() - self.t0
        print("\n" + "=" * 65)
        print("[time summary]")
        for label, elapsed in self.records:
            print(f"  {label:<45s} {elapsed:>8.2f} s")
        print("-" * 65)
        print(f"  {'total':<45s} {total:>8.2f} s ({total / 60.0:.2f} min)")
        print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time odom to COLMAP")
    parser.add_argument("--data_dir", type=str,
                        default="G:/Data/Laser_data/2026-06-22_15-04-26rrrr",
                        help="input data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output COLMAP directory")
    parser.add_argument("--align_mode", type=str, default="none",
                        choices=["mean", "start", "end", "none"],
                        help="how to align image and odom timestamps. "
                             "Default 'none' because extracted mcap images are already "
                             "in the same clock as odom-realtime.csv.")
    parser.add_argument("--odom_path", type=str, default=None,
                        help="path to odometry CSV (default: <data_dir>/odom-realtime.csv). "
                             "Use the manufacturer's refined odom.csv if you want poses matching "
                             "transforms.json/ImgPose.txt.")
    parser.add_argument("--mcap_path", type=str, default=None,
                        help="path to data_raw.mcap (default: <data_dir>/data/data_raw.mcap)")
    parser.add_argument("--axis_align", type=str, default="none",
                        choices=["none", "x180", "y180", "z180"],
                        help="axis alignment between device camera frame and COLMAP camera frame; "
                             "default 'none' keeps the device/COLMAP OpenCV convention (Y down, Z front). "
                             "Use 'x180' to match the manufacturer's transforms.json / 3DGS convention "
                             "(Y up, Z back), which is rotated 180 degrees around X relative to COLMAP.")
    parser.add_argument("--num_points", type=int, default=500000,
                        help="target number of points in the two-stage down-sampling")
    parser.add_argument("--voxel_size", type=float, default=0.5,
                        help="voxel edge length for the second sampling stage")
    parser.add_argument("--random_ratio", type=float, default=0.6,
                        help="fraction of points sampled uniformly at random in the first stage")
    parser.add_argument("--fmt", choices=["txt", "bin", "both"], default="txt",
                        help="COLMAP sparse model output format (default: bin)")
    parser.add_argument("--max_workers", type=int, default=0,
                        help="parallel workers for undistortion; 0 means auto (60%% of CPU cores)")
    parser.add_argument("--undistort_interp", choices=["nearest", "linear", "cubic", "lanczos4"],
                        default="cubic",
                        help="interpolation used by cv2.remap during undistortion "
                             "(default: lanczos4 for best quality; linear is faster)")
    parser.add_argument("--skip_extract", action="store_true",
                        help="skip extracting images from data_raw.mcap "
                             "(assume <output_dir>/cameras/left and right already exist)")
    parser.add_argument("--skip_undistort", action="store_true",
                        help="skip undistortion (assume images/{left,right}/ already populated)")
    parser.add_argument("--skip_pointcloud", action="store_true",
                        help="skip point cloud processing")
    args = parser.parse_args()

    timer = StepTimer()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    sparse_dir = os.path.join(output_dir, "lidar-sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    script_start = time.time()

    timer.start("load calibration & odom")
    # ------------------------------------------------------------------
    # 1. Calibration
    # ------------------------------------------------------------------
    calib_path = os.path.join(data_dir, "info", "calibration.json")
    cams, imu_calib = load_calibration(calib_path)
    print("[calib] loaded cameras:", list(cams.keys()))

    R_il, t_il = build_body_to_lidar(imu_calib)
    print(f"[calib] T_imu_to_lidar t={t_il} (approx identity rotation)")

    # ------------------------------------------------------------------
    # 2. Odom
    # ------------------------------------------------------------------
    odom_path = args.odom_path if args.odom_path is not None else os.path.join(data_dir, "odom-realtime.csv")
    odom_stamps, odom_xyz, odom_quat = load_odom(odom_path)
    print(f"[odom] {len(odom_stamps)} poses, "
          f"freq={1e9/np.median(np.diff(odom_stamps)):.2f} Hz, "
          f"duration={(odom_stamps[-1]-odom_stamps[0])/1e9:.2f} s")
    timer.finish()

    timer.start("extract images from mcap")
    # ------------------------------------------------------------------
    # 3. Extract images from mcap into <output_dir>/cameras/{left,right}
    # ------------------------------------------------------------------
    cameras_dir = os.path.join(output_dir, "fisheye-images")
    mcap_path = args.mcap_path if args.mcap_path is not None else os.path.join(data_dir, "data", "data_raw.mcap")

    if not args.skip_extract:
        extract_images_from_mcap(mcap_path, cameras_dir, time_field="publish_time")
    else:
        print("[extract] skipped (--skip_extract)")
    timer.finish()

    timer.start("filter images by odom range")
    # ------------------------------------------------------------------
    # 4. Load raw timestamps, drop frames outside odom range
    # ------------------------------------------------------------------
    raw_records = {}  # name -> (stamps, paths, names)
    for name in ("left", "right"):
        src_dir = os.path.join(cameras_dir, name)
        stamps, paths, names = parse_image_stamps(src_dir)
        raw_records[name] = (stamps, paths, names)
        print(f"[{name}] extracted {len(stamps)} images from mcap")

    # Compute offset on raw stamps (default 0)
    all_raw_stamps = np.concatenate([raw_records[n][0] for n in ("left", "right")])
    offset = compute_time_offset(all_raw_stamps, odom_stamps, mode=args.align_mode)
    print(f"\n[align] image vs odom offset = {offset/1e9:.3f} s "
          f"({offset/1e9/86400:.2f} days), mode={args.align_mode}")
    print(f"[align] image range: {all_raw_stamps[0]/1e9:.3f} -> {all_raw_stamps[-1]/1e9:.3f}")
    print(f"[align] odom  range: {odom_stamps[0]/1e9:.3f} -> {odom_stamps[-1]/1e9:.3f}")

    # Filter out images that would query outside odom range
    valid_records = {}
    for name in ("left", "right"):
        stamps, paths, names = raw_records[name]
        valid = (stamps - offset >= odom_stamps[0]) & (stamps - offset <= odom_stamps[-1])
        n_dropped = len(stamps) - int(valid.sum())
        if n_dropped > 0:
            print(f"[{name}] dropping {n_dropped} image(s) outside odom range")
        if valid.sum() == 0:
            raise RuntimeError(f"No valid images left for camera {name} after odom range filter")
        valid_records[name] = (stamps[valid], paths[valid], names[valid])
    timer.finish()

    timer.start("undistort images")
    # ------------------------------------------------------------------
    # 5. Undistortion + COLMAP camera definitions
    # ------------------------------------------------------------------
    image_records = []  # (camera_id, camera_name, stamp, image_name, rel_path)
    cameras_bin = {}

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interp = interp_map[args.undistort_interp]
    if args.max_workers <= 0:
        actual_workers = max(1, int((os.cpu_count() or 1) * _DEFAULT_WORKER_RATIO))
    else:
        actual_workers = args.max_workers
    print(f"[undistort] interpolation={args.undistort_interp}, max_workers={actual_workers}")

    for cam_id, name in enumerate(["left", "right"], start=1):
        cam = cams[name]
        src_dir = os.path.join(cameras_dir, name)
        out_img_dir = os.path.join(output_dir, "undistorted-images", name)
        os.makedirs(out_img_dir, exist_ok=True)

        stamps, paths, names = valid_records[name]

        # Always use the calibrated camera matrix as the target of undistortion.
        target_size = (int(cam["width"]), int(cam["height"]))
        target_intrinsic = cam["intrinsic"].copy()        
        target_intrinsic["cx"] = cam["width"] / 2
        target_intrinsic["cy"] = cam["height"] / 2
        target_K = make_K(target_intrinsic)

        if args.skip_undistort:
            # Assume images already exist in output_dir/images/{name}/
            # and have the same filenames as the source.
            intrinsic = {
                "fl_x": target_intrinsic["fl_x"],
                "fl_y": target_intrinsic["fl_y"],
                "cx": target_intrinsic["cx"],
                "cy": target_intrinsic["cy"],
                "width": cam["width"],
                "height": cam["height"],
            }
            print(f"[{name}] skip undistort, using {len(names)} existing images")
            print(f"[{name}] using OPENCV intrinsics: fx={intrinsic['fl_x']:.3f} fy={intrinsic['fl_y']:.3f} "
                  f"cx={intrinsic['cx']:.3f} cy={intrinsic['cy']:.3f} ({intrinsic['width']}x{intrinsic['height']})")
        else:
            intrinsic, stamps, names = undistort_camera(
                name, cam, src_dir, out_img_dir,
                target_size=target_size,
                target_K=target_K, max_workers=actual_workers,
                interp=interp,
                stamps=stamps, paths=paths, names=names
            )
            print(f"after undistort, intrinsics: fx={intrinsic['fl_x']:.3f} fy={intrinsic['fl_y']:.3f} "
                  f"cx={intrinsic['cx']:.3f} cy={intrinsic['cy']:.3f}")

        # Images are already undistorted, so keep calibration intrinsics and
        # set the OPENCV distortion coefficients to zero.
        params = np.array([
            intrinsic["fl_x"], intrinsic["fl_y"],
            intrinsic["cx"], intrinsic["cy"],
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float64)
        cameras_bin[cam_id] = Camera(
            id=cam_id,
            model="OPENCV",
            width=intrinsic["width"],
            height=intrinsic["height"],
            params=params,
        )

        for s, n in zip(stamps, names):
            image_records.append((cam_id, name, s, n, f"{name}/{n}"))
    timer.finish()

    timer.start("interpolate poses")
    # ------------------------------------------------------------------
    # 6. Interpolate poses and write COLMAP images.bin
    # ------------------------------------------------------------------
    images_bin = {}
    image_id = 1
    skipped = 0
    for cam_id, name, img_stamp, img_name, rel_path in sorted(image_records, key=lambda x: (x[2], x[0])):
        t_query = img_stamp - offset  # map image time into odom clock

        R_wi, t_wi = interpolate_pose(t_query, odom_stamps, odom_xyz, odom_quat)
        if R_wi is None:
            print(f"[warn] {img_name}: timestamp out of odom range, skipped")
            skipped += 1
            continue

        # calibration.json stores transform_from_lidar = camera->lidar.
        # We need lidar->camera, so invert it first.
        R_cl = np.array(cams[name]["transform_from_lidar"]["rotation"], dtype=np.float64)
        t_cl = np.array(cams[name]["transform_from_lidar"]["position"], dtype=np.float64)
        R_lc, t_lc = invert_pose(R_cl, t_cl)

        # world -> camera
        R_wc, t_wc = build_camera_pose(R_wi, t_wi, R_il, t_il, R_lc, t_lc)
        R_wc, t_wc = align_to_colmap_axis(R_wc, t_wc, args.axis_align)
        R_cw, t_cw = invert_pose(R_wc, t_wc)
        qvec, tvec = pose_to_colmap_qt(R_cw, t_cw)

        images_bin[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=cam_id,
            name=rel_path,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros(0, dtype=np.int64),
        )
        image_id += 1

    print(f"[images] registered {len(images_bin)} images, skipped {skipped}")
    timer.finish()

    timer.start("process point cloud")
    # ------------------------------------------------------------------
    # 7. Point cloud -> sparse/0/points3D.ply
    # ------------------------------------------------------------------
    points3D_pcd = None
    if not args.skip_pointcloud:
        las_path = os.path.join(data_dir, "colorized-realtime.las")
        points3D_ply = os.path.join(sparse_dir, "points3D.ply")
        points3D_pcd = process_point_cloud(
            las_path,
            points3D_ply,
            voxel_size=args.voxel_size,
            random_ratio=args.random_ratio,
            target_num_points=args.num_points,
        )
    timer.finish()

    timer.start("write COLMAP model")
    # ------------------------------------------------------------------
    # 8. Write COLMAP model (text/binary/both)
    # ------------------------------------------------------------------
    points3D = build_points3D_dict(points3D_pcd) if points3D_pcd is not None else {}
    if args.fmt in ("bin", "both"):
        write_model(cameras_bin, images_bin, points3D, sparse_dir, ext=".bin")
    if args.fmt in ("txt", "both"):
        write_model(cameras_bin, images_bin, points3D, sparse_dir, ext=".txt")
    print(f"[colmap] wrote sparse model format={args.fmt} to {sparse_dir}")
    timer.finish()

    timer.summary()
    elapsed = time.time() - script_start
    print(f"[done] COLMAP model ready at: {output_dir}")
    print(f"[timer] Total elapsed time: {elapsed:.2f} s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
