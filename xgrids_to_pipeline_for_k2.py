#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgrids_to_pipeline.py
=====================
把 XGRIDS（其域）手持扫描仪数据适配成 run_laser_sfm_pipeline.py 认识的 data_dir 布局。

输入（一次采集目录，如 0709-001_2026-07-09-141245/）：
    <data_dir>/<name>.xbin              XBAG 容器（视频流 + 标定包）
    <data_dir>/project_data/poses.csv  LIO 实时轨迹（秒, x y z qx qy qz qw）
    <data_dir>/map.las                 SLAM 彩色点云地图

输出（--out_dir，默认 <data_dir>/pipeline_input）：
    cameras/left/<ns>.jpg              cam0 鱼眼帧（JPEG，文件名=纳秒时间戳）
    cameras/right/<ns>.jpg           cam1 鱼眼帧
    info/calibration.json            kb4(=OpenCV fisheye) 内参 + transform_from_lidar
    odom-realtime.csv                poses.csv 时间戳 ×1e9（纳秒）
    colorized-realtime.las           map.las 硬链接/复制
    frames_ts.csv                    每帧时间戳对照（cam, frame_idx, ts_ns）
    validation/                      las 点云 → 鱼眼图投影叠加（外参方向目检）
    step1/                           可选：直接调用 real_time_to_colmap.py 的输出

外参方向说明
------------
其域 extrinsic_camera_lidar.yaml 的 transform 方向无文档，本脚本对两种假设
(A: transform=T_lc 相机在激光系中的位姿; B: 为其逆) 都生成投影验证图，
人工目检后通过 --extrinsic_mode a|b 定版（默认 b，先跑验证再决定）。

相机1 的外参 = 相机0外参 与 camera.yaml 中 camera_1.camera_pose 的串联，
串联顺序同样给两种假设（--cam1_chain left|right）。

随后接流水线：
    python run_laser_sfm_pipeline.py \
        --data_dir <out_dir> --output_dir <out_dir>/laser_sfm \
        --colmap_exe <colmap.exe> --resume
（本脚本 --run_step1 会先把帧摆到 01_lidar_colmap/cameras 并写好 .done 标记）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from unicode_paths import imread as unicode_imread, imwrite as unicode_imwrite


# ---------------------------------------------------------------------------
# xbag reader (import from extract_xbag.py)
# ---------------------------------------------------------------------------
def load_xbag_module(xbag_script: Path):
    spec = importlib.util.spec_from_file_location("extract_xbag", str(xbag_script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# minimal yaml value scraping (configs are simple flat files)
# ---------------------------------------------------------------------------
def yaml_key_list(text: str, key: str):
    """Return list of floats for `key:` given as inline [..] or '- v' items."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*\[(.*?)\]", text, re.M | re.S)
    if m:
        body = m.group(1)
        # stop at first ']' already captured; may span lines
        return [float(x) for x in body.replace("\n", " ").split(",") if x.strip()]
    m = re.search(rf"^\s*{re.escape(key)}:\s*\n((?:\s*-\s*[^\n]+\n?)+)", text, re.M)
    if not m:
        return None
    vals = []
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*-\s*(\S+)", line)
        if mm:
            vals.append(float(mm.group(1)))
    return vals

#
# def yaml_key_scalar(text: str, key: str):
#     m = re.search(rf"^\s*{re.escape(key)}:\s*([^\n\[]+)$", text, re.M)
#     return m.group(1).strip() if m else None
#
#
# def parse_camera_yaml(text: str):
#     cams = {}
#     for sec in ("camera_0", "camera_1"):
#         m = re.search(rf"^{sec}:\n(.*?)(?=^camera_1:|^calibrated:|\Z)", text, re.M | re.S)
#         body = m.group(1)
#         intr = yaml_key_list(body, "intrinsic")
#         dist = yaml_key_list(body, "distortion")
#         pose = yaml_key_list(body, "camera_pose")
#         w = int(float(yaml_key_scalar(body, "image_width")))
#         h = int(float(yaml_key_scalar(body, "image_height")))
#         cams[sec] = dict(intrinsic=intr, distortion=dist, camera_pose=pose, width=w, height=h)
#     return cams

def yaml_key_scalar(text: str, key: str):
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^\n\[]+)$", text, re.M)
    return m.group(1).strip() if m else None
def parse_camera_yaml(text: str):
    """
    从 camera.yaml 解析所有相机配置。
    返回: dict {camera_id: {intrinsic, distortion, camera_pose, width, height, camera_model}}
    """
    cams = {}
    # 匹配所有 camera_X 块
    pattern = r'^(camera_\d+):\n(.*?)(?=^camera_\d+:|^calibrated:|\Z)'
    for match in re.finditer(pattern, text, re.M | re.S):
        sec = match.group(1)
        body = match.group(2)

        # 提取 camera_model（如果有）
        model_match = re.search(r'camera_model:\s*(\w+)', body)
        camera_model = model_match.group(1) if model_match else "kb4"

        intr = yaml_key_list(body, "intrinsic")
        dist = yaml_key_list(body, "distortion")
        pose = yaml_key_list(body, "camera_pose")
        w = int(float(yaml_key_scalar(body, "image_width")))
        h = int(float(yaml_key_scalar(body, "image_height")))

        cams[sec] = dict(
            intrinsic=intr,
            distortion=dist,
            camera_pose=pose,
            width=w,
            height=h,
            camera_model=camera_model
        )
    return cams
def get_camera_list_from_yaml(cam_yaml_text):
    """从 camera.yaml 解析相机列表，返回 [(cam_id, name, model), ...]"""
    import re
    cams = []
    # 查找所有 camera_X 块
    pattern = r'^(camera_\d+):\n(.*?)(?=^camera_\d+:|^calibrated:|\Z)'
    for match in re.finditer(pattern, cam_yaml_text, re.M | re.S):
        cam_id = match.group(1)
        body = match.group(2)
        # 提取 camera_model
        model_match = re.search(r'camera_model:\s*(\w+)', body)
        model = model_match.group(1) if model_match else "kb4"
        cams.append((cam_id, model))
    return cams


def sort_yaml_content(content: bytes) -> bytes:
    """Sort YAML mapping keys recursively while preserving sequence order."""
    parsed = yaml.safe_load(content.decode("utf-8"))
    sorted_text = yaml.safe_dump(
        parsed, allow_unicode=True, sort_keys=True, default_flow_style=False
    )
    return sorted_text.encode("utf-8")

# ---------------------------------------------------------------------------
# xbin walking: frames + config files
# ---------------------------------------------------------------------------
def walk_xbin(xbag_mod, xbin_path: Path):
    """Yield dicts: video frames (cam, ts_ns, au bytes), configs {name: bytes}."""
    Xbag = xbag_mod.Xbag
    pb = xbag_mod.pb
    x = Xbag(str(xbin_path))
    _, front_end = x.header_msg()
    rec_start = x.first_record_off(front_end)
    video_rec_count = 0
    frames = []  # (cam, ts_ns, au_bytes)
    configs = {}
    # 添加一个集合来记录所有出现的 topic
    seen_topics = set()
    for rec in x.records(rec_start):
        topic = rec["topic"]
        seen_topics.add(topic)
        if rec["topic"] == 5:
            envelopes = [env for fld, wire, env in pb(rec["payload"]) if fld == 1 and wire == 2]
            rec_is_video = False
            parsed = []
            for env in envelopes:
                inner = pb(env)
                vdata = next((v2 for f2, w2, v2 in inner if f2 == 5 and w2 == 2), None)
                if vdata is not None and vdata[:4] == b"\x00\x00\x00\x01":
                    hdr = next((v2 for f2, w2, v2 in inner if f2 == 1), None)
                    hf = dict((f2, v2) for f2, w2, v2 in pb(hdr) if w2 == 0) if hdr else {}
                    parsed.append((hf.get(2), vdata))  # ts in 0.5us units
                    rec_is_video = True
            if rec_is_video:
                cam = video_rec_count % 3
                video_rec_count += 1
                for ts_half_us, au in parsed:
                    frames.append((cam, int(ts_half_us) * 500, au))  # -> ns
        elif rec["topic"] == 6 or rec["topic"] == 7:
            # print(f"[debug] Found topic 6")
            for fld, wire, item in pb(rec["payload"]):
                if fld != 1 or wire != 2:
                    continue
                name = content = None
                for f2, w2, v2 in pb(item):
                    if f2 == 2 and w2 == 2:
                        name = v2.decode("utf-8", "replace")
                        # print(f"[debug] Config name: {name}")
                    elif f2 == 3 and w2 == 2:
                        content = v2
                if name and content:
                    configs[os.path.basename(name)] = content
                    # print(f"[debug] Added config: {os.path.basename(name)}")
        # else:
        #
        #     print("rec[topic] != 5或6")
    frames.sort(key=lambda f: (f[0], f[1]))
    print(f"[debug] Config keys: {list(configs.keys())}")
    return frames, configs

# def walk_xbin(xbag_mod, xbin_path: Path):
#     """Yield dicts: video frames (cam, ts_ns, au bytes), configs {name: bytes}."""
#     Xbag = xbag_mod.Xbag
#     pb = xbag_mod.pb
#     x = Xbag(str(xbin_path))
#     _, front_end = x.header_msg()
#     rec_start = x.first_record_off(front_end)
#     video_rec_count = 0
#     frames = []  # (cam, ts_ns, au_bytes)
#     configs = {}
#
#     # 添加一个集合来记录所有出现的 topic
#     seen_topics = set()
#     topic_5_count = 0
#     topic_6_count = 0
#
#     for rec in x.records(rec_start):
#         topic = rec["topic"]
#         seen_topics.add(topic)
#
#         if topic == 5:
#             topic_5_count += 1
#             if topic_5_count <= 3:  # 只打印前3个 topic 5 的详细信息
#                 print(f"[debug] Topic 5 #{topic_5_count}, payload length: {len(rec['payload'])}")
#                 try:
#                     # 尝试解析第一个 topic 5 的结构
#                     for fld, wire, item in pb(rec["payload"]):
#                         if fld == 1 and wire == 2:
#                             print(f"[debug]   Envelope found, len={len(item)}")
#                             # 解析 envelope
#                             for f2, w2, v2 in pb(item):
#                                 if f2 == 2 and w2 == 0:  # timestamp
#                                     print(f"[debug]     ts: {v2}")
#                                 elif f2 == 5 and w2 == 2:  # video data
#                                     if v2 and len(v2) > 10:
#                                         print(f"[debug]     video data len={len(v2)}, start={v2[:8].hex()}")
#                                         # 检查是否以 H.264 NAL 头开始
#                                         is_h264 = v2[:4] == b'\x00\x00\x00\x01'
#                                         print(f"[debug]     is H.264 NAL: {is_h264}")
#                                 elif f2 == 1 and w2 == 0:  # timestamp field
#                                     print(f"[debug]     ts field: {v2}")
#                     print(f"[debug]   ---")
#                 except Exception as e:
#                     print(f"[debug]   Parse error: {e}")
#
#             # 原有的视频处理代码（保留但不修改）
#             envelopes = [env for fld, wire, env in pb(rec["payload"]) if fld == 1 and wire == 2]
#             rec_is_video = False
#             parsed = []
#             for env in envelopes:
#                 inner = pb(env)
#                 vdata = next((v2 for f2, w2, v2 in inner if f2 == 5 and w2 == 2), None)
#                 if vdata is not None and vdata[:4] == b"\x00\x00\x00\x01":
#                     hdr = next((v2 for f2, w2, v2 in inner if f2 == 1), None)
#                     hf = dict((f2, v2) for f2, w2, v2 in pb(hdr) if w2 == 0) if hdr else {}
#                     parsed.append((hf.get(2), vdata))  # ts in 0.5us units
#                     rec_is_video = True
#             if rec_is_video:
#                 cam = video_rec_count % 2
#                 video_rec_count += 1
#                 for ts_half_us, au in parsed:
#                     frames.append((cam, int(ts_half_us) * 500, au))  # -> ns
#
#         elif topic == 6 or topic == 7 or topic == 200:
#             # 只打印非视频 topic 的数量
#             if topic == 6:
#                 topic_6_count += 1
#                 if topic_6_count <= 2:
#                     print(f"[debug] Topic {topic} found, payload length: {len(rec['payload'])}")
#                     # 尝试查找 yaml 内容
#                     try:
#                         text = rec['payload'].decode('utf-8', 'ignore')
#                         if 'camera.yaml' in text or 'extrinsic' in text:
#                             print(f"[debug]   Found yaml content in topic {topic}!")
#                             # 尝试提取 yaml 内容
#                             import re
#                             match = re.search(r'camera\.yaml[^\x00]*?camera_0:', text)
#                             if match:
#                                 print(f"[debug]   camera.yaml found at position {match.start()}")
#                     except:
#                         pass
#         else:
#             # 其他 topic 只计数，不打印
#             pass
#
#     print(f"[debug] Topic counts: topic5={topic_5_count}, topic6={topic_6_count}")
#     print(f"[debug] All topics seen: {seen_topics}")
#     print(f"[debug] Video frames decoded: {len(frames)}")
#     frames.sort(key=lambda f: (f[0], f[1]))
#     return frames, configs
# ---------------------------------------------------------------------------
# frame decode: one ffmpeg pass per camera, then rename to ns timestamps
# ---------------------------------------------------------------------------
def decode_camera_frames(frames, cam, ffmpeg: str, work_dir: Path, out_dir: Path,
                         jpeg_quality: int = 2, max_frames: int = 0):
    cam_frames = [f for f in frames if f[0] == cam]
    cam_frames.sort(key=lambda f: f[1])
    if max_frames > 0:
        cam_frames = cam_frames[:max_frames]
    tmp = work_dir / f"_cam{cam}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    stream = tmp / "stream.h264"
    with open(stream, "wb") as w:
        for _, _, au in cam_frames:
            w.write(au)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(stream),
         "-q:v", str(jpeg_quality), str(tmp / "f_%05d.jpg")],
        check=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_table = []
    jpgs = sorted(tmp.glob("f_*.jpg"))
    assert len(jpgs) == len(cam_frames), \
        f"decoded {len(jpgs)} != frames {len(cam_frames)} for cam{cam}"
    for (ts_ns, _), jpg in zip([(f[1], None) for f in cam_frames], jpgs):
        dst = out_dir / f"{ts_ns}.jpg"
        shutil.move(str(jpg), str(dst))
        ts_table.append((cam, jpg.stem, ts_ns))
    shutil.rmtree(tmp)
    return ts_table


# ---------------------------------------------------------------------------
# extrinsic assembly
# ---------------------------------------------------------------------------
def mat4(vals):
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def inv4(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def build_T_lc(extrinsic_vals, cam1_pose_vals, cam_idx: int,
               extrinsic_mode: str, cam1_chain: str):
    """Return T_lc: camera pose in lidar frame (p_lidar = T_lc @ p_cam)."""
    T_ext = mat4(extrinsic_vals)
    T_lc0 = T_ext if extrinsic_mode == "a" else inv4(T_ext)
    if cam_idx == 0:
        return T_lc0
    T_10 = mat4(cam1_pose_vals)  # camera_1 pose in camera_0 frame (assumed)
    if cam1_chain == "left":
        return T_lc0 @ T_10
    return T_lc0 @ inv4(T_10)


def T_lc_to_transform_from_lidar(T_lc):
    """pipeline calibration.json wants p_cam = T @ p_lidar (inverse of T_lc)."""
    T_cl = inv4(T_lc)
    return {
        "rotation": T_cl[:3, :3].tolist(),
        "position": T_cl[:3, 3].tolist(),
    }


# ---------------------------------------------------------------------------
# validation: project las points into fisheye frames
# ---------------------------------------------------------------------------
def load_poses(poses_csv: Path):
    data = np.loadtxt(str(poses_csv), delimiter=",", comments="#")
    ts = data[:, 0] * 1e9  # ns
    return ts, data[:, 1:4], data[:, 4:8]


def interp_pose(t_query, ts, xyz, quat):
    from scipy.spatial.transform import Slerp, Rotation as R
    idx = np.searchsorted(ts, t_query)
    if idx == 0 or idx >= len(ts):
        idx = min(max(idx, 1), len(ts) - 1)
    x = np.interp(t_query, ts, xyz[:, 0])
    y = np.interp(t_query, ts, xyz[:, 1])
    z = np.interp(t_query, ts, xyz[:, 2])
    slerp = Slerp(ts, R.from_quat(quat))
    r = slerp([t_query])[0]
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def kb4_project(pts_cam, K, D):
    """Project Nx3 camera-frame points with OpenCV fisheye (kb4) model."""
    import cv2
    pts = pts_cam.reshape(-1, 1, 3).astype(np.float64)
    uv, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2)


def make_validation_overlays(las_path, poses_csv, frames, cams_cfg, extrinsic_vals,
                             cam1_pose_vals, extrinsic_mode, cam1_chain,
                             out_dir: Path, frame_index: int, n_points: int = 60000,
                             axis_align: str = "z180"):
    import cv2
    import laspy

    # 与 real_time_to_colmap.py 的 align_to_colmap_axis 保持一致（右乘相机系）
    AXIS_FIX = {"none": np.eye(4),
                "x180": np.diag([1., -1., -1., 1.]),
                "y180": np.diag([-1., 1., -1., 1.]),
                "z180": np.diag([-1., -1., 1., 1.])}
    fix = AXIS_FIX[axis_align]

    ts, xyz, quat = load_poses(poses_csv)
    las = laspy.read(str(las_path))
    total = len(las.x)
    stride = max(1, total // n_points)
    P = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)[::stride]
    rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64)[::stride]
    if rgb.max() > 255:
        rgb /= 256.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    results = {}
    for cam_idx, cam_name in ((0, "left"), (1, "right")):
        cam_frames = [f for f in frames if f[0] == cam_idx]
        if not cam_frames:
            continue
        ts_ns = cam_frames[min(frame_index, len(cam_frames) - 1)][1]
        T_wl = interp_pose(ts_ns, ts, xyz, quat)  # lidar pose in world
        T_lc = build_T_lc(extrinsic_vals, cam1_pose_vals, cam_idx,
                          extrinsic_mode, cam1_chain) @ fix
        T_cw = inv4(T_wl @ T_lc)  # world -> camera
        Pc = (T_cw[:3, :3] @ P.T).T + T_cw[:3, 3]
        front = Pc[:, 2] > 0.1

        cfg = cams_cfg[f"camera_{cam_idx}"]
        fx, fy, cx, cy = cfg["intrinsic"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        D = np.array(cfg["distortion"], dtype=np.float64)
        uv = kb4_project(Pc[front], K, D)
        W, H = cfg["width"], cfg["height"]
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        img_path = out_dir / "cameras" / cam_name / f"{ts_ns}.jpg"
        img = unicode_imread(img_path)
        vis = cv2.resize(img, (1000, 750)) if img is not None else \
            np.zeros((750, 1000, 3), np.uint8)
        sx, sy = 1000.0 / W, 750.0 / H
        pts_uv = uv[inb]
        pts_rgb = rgb[front][inb]
        for (u, v), c in zip(pts_uv[::2], pts_rgb[::2]):
            cv2.circle(vis, (int(u * sx), int(v * sy)), 1,
                       (int(c[2]), int(c[1]), int(c[0])), -1)
        n_in = int(inb.sum())
        n_front = int(front.sum())
        results[cam_name] = dict(frame_ts=ts_ns, points_front=n_front,
                                 points_in_image=n_in, total=len(P))
        cv2.putText(vis, f"{cam_name} mode={extrinsic_mode}/{cam1_chain} "
                         f"front={n_front}/{len(P)} inimg={n_in}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        validation_path = (
            out_dir / "validation" /
            f"proj_{cam_name}_{extrinsic_mode}_{cam1_chain}.jpg"
        )
        if not unicode_imwrite(validation_path, vis):
            raise RuntimeError(f"Cannot write validation image: {validation_path}")
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="XGRIDS data -> laser-sfm pipeline data_dir")
    ap.add_argument("--data_dir", required=True, help="XGRIDS capture dir (xbin/map.las/project_data)")
    ap.add_argument("--output_dir", default=None, help="output pipeline data_dir (default: <data_dir>/pipeline_input)")
    ap.add_argument("--xbin", default=None, help="xbin path (default: single .xbin in data_dir)")
    ap.add_argument("--xbag_script", default=r"utils\extract_xbag.py",
                    help="path to extract_xbag.py (Xbag reader)")
    ap.add_argument("--ffmpeg", default=r"tools\ffmpeg-9.0.1\bin\ffmpeg.exe")
    ap.add_argument("--jpeg_quality", type=int, default=1, help="ffmpeg -q:v (default 2)")
    ap.add_argument("--max_frames", type=int, default=0, help="limit frames per camera (0=all)")
    ap.add_argument("--extrinsic_mode", choices=["a", "b"], default="b",
                    help="a: extrinsic yaml = T_lc as-is; b: invert it (default b)")
    ap.add_argument("--cam1_chain", choices=["left", "right"], default="left",
                    help="cam1 extrinsic chain: T_lc0@pose (left) or T_lc0@inv(pose) (right)")
    ap.add_argument("--validate", action="store_true",
                    help="render las->fisheye projection overlays into validation/ "
                         "(runs both extrinsic modes regardless of --extrinsic_mode)")
    ap.add_argument("--validate_frame", type=int, default=100)
    ap.add_argument("--run_step1", action="store_true",
                    help="after adapting, invoke real_time_to_colmap.py --skip_extract "
                         "into <pipeline_output>/01_lidar_colmap and write .done marker")
    ap.add_argument("--pipeline_output", default=None,
                    help="pipeline output root for --run_step1 (default: <out_dir>/laser_sfm)")
    # ap.add_argument("--pipeline_root", default=r"G:\project\sfm-pipeline-dev_wl-20260629")
    ap.add_argument("--pipeline_root", default=r"E:\pyDevelop\laser_data_sfm")
    ap.add_argument("--step1_python", default=sys.executable)
    ap.add_argument("--axis_align", default="z180",
                    choices=["none", "x180", "y180", "z180"],
                    help="相机系轴向对齐。XGRIDS K1 实测需要 z180（绕光轴 roll 180°），"
                         "否则投影内容上下颠倒（高处点落到图像下半部）")
    ap.add_argument("--num_points", type=int, default=500000)
    ap.add_argument("--keep_radius", type=float, default=5.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (data_dir / "pipeline_input")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "info").mkdir(exist_ok=True)
    (out_dir / "cameras" / "left").mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras" / "right").mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras" / "front").mkdir(parents=True, exist_ok=True)

    xbin = Path(args.xbin) if args.xbin else next(data_dir.glob("*.xbin"))
    poses_csv = data_dir / "project_data" / "poses.csv"
    las_src = data_dir / "map.las"
    for p in (xbin, poses_csv, las_src):
        if not p.exists():
            raise SystemExit(f"[error] missing: {p}")

    ffmpeg = args.ffmpeg if Path(args.ffmpeg).exists() else shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("[error] ffmpeg not found")

    # ---- 1. walk xbin ------------------------------------------------------
    print(f"[xbin] walking {xbin.name} ...")
    xbag_mod = load_xbag_module(Path(args.xbag_script))
    frames, configs = walk_xbin(xbag_mod, xbin)
    n0 = sum(1 for f in frames if f[0] == 0)
    n1 = sum(1 for f in frames if f[0] == 1)
    n2 = sum(1 for f in frames if f[0] == 2)
    print(f"[xbin] video frames: cam0={n0} cam1={n1} cam2={n2}; configs: {sorted(configs)}")

    cfg_dir = out_dir / "xbin_configs"
    cfg_dir.mkdir(exist_ok=True)
    for name, content in configs.items():
        output_content = (
            sort_yaml_content(content)
            if Path(name).suffix.lower() in {".yaml", ".yml"}
            else content
        )
        (cfg_dir / name).write_bytes(output_content)

    cam_yaml = configs["camera.yaml"].decode("utf-8", "replace")
    cams_cfg = parse_camera_yaml(cam_yaml)
    ext_cl = yaml_key_list(configs["extrinsic_camera_lidar.yaml"]
                           .decode("utf-8", "replace"), "transform")
    cam1_pose = cams_cfg["camera_1"]["camera_pose"]

    # ---- 2. decode frames ---------------------------------------------------
    ts_table = []
    for cam, cam_name in ((1 ,"left"), (2, "right"), (0, "front")): # 图像对应的相机索引，front是 2
        print(f"[decode] cam{cam} -> cameras/{cam_name} ...")
        ts_table += decode_camera_frames(
            frames, cam, ffmpeg, out_dir, out_dir / "cameras" / cam_name,
            jpeg_quality=args.jpeg_quality, max_frames=args.max_frames)
    with open(out_dir / "frames_ts.csv", "w") as w:
        w.write("cam,frame_tag,ts_ns\n")
        for cam, tag, ts_ns in ts_table:
            w.write(f"{cam},{tag},{ts_ns}\n")
    print(f"[decode] wrote {len(ts_table)} jpgs")

    # ---- 3. calibration.json ------------------------------------------------
    calib = {"cameras": [], "imu": []}
    for cam_idx, cam_name in ((0, "left"), (1, "right"), (2, "front")):
        cfg = cams_cfg[f"camera_{cam_idx}"]  # 相机内参front的ID是2，但是
        fx, fy, cx, cy = cfg["intrinsic"]
        cam_pose = cfg["camera_pose"]
        T_lc = build_T_lc(ext_cl, cam_pose, cam_idx, args.extrinsic_mode, args.cam1_chain)
        cam_model = cfg.get("camera_model", "kb4")  # 默认使用 kb4，如果没有指定
        if cam_model == "kb4":
            cam_model = "OPENCV_FISHEYE"
            distortion_model = ("k1", "k2", "k3", "k4")
        else:
            cam_model = "OPENCV"
            distortion_model = ("k1", "k2", "p1", "p2", "k3")
        calib["cameras"].append({
            "name": cam_name,
            "id": cam_idx,
            "width": cfg["width"],
            "height": cfg["height"],
            "intrinsic": {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy},
            "distortion": {"model": cam_model,
                           "params": dict(zip(distortion_model, cfg["distortion"]))},
            "transform_from_lidar": T_lc_to_transform_from_lidar(T_lc),
        })
    with open(out_dir / "info" / "calibration.json", "w") as w:
        json.dump(calib, w, indent=2)
    print(f"[calib] wrote calibration.json (extrinsic_mode={args.extrinsic_mode}, "
          f"cam1_chain={args.cam1_chain})")

    # ---- 4. odom + las ------------------------------------------------------
    poses = np.loadtxt(str(poses_csv), delimiter=",", comments="#")
    poses_ns = poses.copy()
    poses_ns[:, 0] *= 1e9
    np.savetxt(str(out_dir / "odom-realtime.csv"), poses_ns,
               delimiter=",", fmt=["%.0f", "%.6f", "%.6f", "%.6f",
                                   "%.8f", "%.8f", "%.8f", "%.8f"])
    print(f"[odom] {len(poses)} poses -> odom-realtime.csv (ns)")

    las_dst = out_dir / "colorized-realtime.las"
    if not las_dst.exists():
        try:
            os.link(str(las_src), str(las_dst))
            print("[las] hardlinked map.las")
        except OSError:
            shutil.copy2(str(las_src), str(las_dst))
            print("[las] copied map.las")

    # ---- 5. validation overlays (both extrinsic hypotheses) -----------------
    if args.validate:
        (out_dir / "validation").mkdir(exist_ok=True)
        for mode in ("a", "b"):
            res = make_validation_overlays(
                las_dst, poses_csv, frames, cams_cfg, ext_cl, cam1_pose,
                mode, args.cam1_chain, out_dir, args.validate_frame,
                axis_align=args.axis_align)
            print(f"[validate] mode={mode}: {res}")
        print(f"[validate] see {out_dir / 'validation'} — pick the mode where "
              f"points hug the surfaces, then rerun with --extrinsic_mode")

    # ---- 6. summary ---------------------------------------------------------
    summary = dict(
        xbin=str(xbin), frames=dict(cam0=n0, cam1=n1),
        poses=len(poses),
        extrinsic_mode=args.extrinsic_mode, cam1_chain=args.cam1_chain,
        calib=cams_cfg,
        extrinsic_camera_lidar=ext_cl,
    )
    with open(out_dir / "adapter_summary.json", "w") as w:
        json.dump(summary, w, indent=2, default=str)

    # ---- 7. optional: run pipeline step1 ------------------------------------
    if args.run_step1:
        pipeline_output = Path(args.pipeline_output).resolve() if args.pipeline_output \
            else (out_dir / "laser_sfm")
        step1_out = pipeline_output / "01_lidar_colmap"
        step1_out.mkdir(parents=True, exist_ok=True)
        # stage frames where real_time_to_colmap.py --skip_extract expects them
        for cam_name in ("left", "right"):
            dst = step1_out / "cameras" / cam_name
            dst.mkdir(parents=True, exist_ok=True)
            src = out_dir / "cameras" / cam_name
            for jpg in src.glob("*.jpg"):
                tgt = dst / jpg.name
                if not tgt.exists():
                    try:
                        os.link(str(jpg), str(tgt))
                    except OSError:
                        shutil.copy2(str(jpg), str(tgt))
        cmd = [
            args.step1_python,
            str(Path(args.pipeline_root) / "real_time_to_colmap.py"),
            "--data_dir", str(out_dir),
            "--output_dir", str(step1_out),
            "--undistort_mode", "fixed",
            "--fmt", "txt",
            "--num_points", str(args.num_points),
            "--keep_radius", str(args.keep_radius),
            "--axis_align", args.axis_align,
            "--skip_extract",
        ]
        print(f"[step1] {' '.join(cmd)}")
        ret = subprocess.call(cmd, cwd=args.pipeline_root)
        if ret == 0:
            (pipeline_output / ".step1_lidar_colmap.done").touch()
            print("[step1] done; continue with:")
            print(f'  python "{Path(args.pipeline_root) / "run_laser_sfm_pipeline.py"}" '
                  f'--data_dir "{out_dir}" --output_dir "{pipeline_output}" --resume '
                  f'--colmap_exe "<colmap.exe>"')
        else:
            raise SystemExit(f"[error] step1 exited with {ret}")

    print(f"\n[done] pipeline data_dir ready: {out_dir}")


if __name__ == "__main__":
    main()
