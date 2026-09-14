#!/usr/bin/env python3
"""Convert post-processed point clouds and image poses to a COLMAP model.

Expected input directory contents:
    colorized.las
    ImgPose.txt
    calibration.json

ImgPose.txt is expected to contain rows in the form::

    image_path x y z roll pitch yaw qx qy qz qw timestamp_seconds

The pose in ImgPose.txt is camera-to-world. COLMAP stores world-to-camera
rotation and translation, so the pose is inverted before writing the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from lidar_data_parser import two_stage_point_cloud_sample
from read_write_model import Camera, Image, Point3D, write_model


def load_calibration(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    cameras = {}
    for camera in calibration.get("cameras", []):
        name = camera.get("name")
        if name in {"left", "right"}:
            cameras[name] = camera
    if not cameras:
        raise ValueError(f"No left/right cameras found in {path}")
    return cameras


def parse_pose_file(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    poses = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            if fields[0].lower() in {"index", "image", "image_path", "file_path"}:
                continue
            if len(fields) < 11:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 11 columns, got {len(fields)}"
                )
            name = fields[0].replace("\\", "/")
            try:
                position = np.asarray([float(value) for value in fields[1:4]])
                quaternion = np.asarray([float(value) for value in fields[7:11]])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric pose") from error
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                raise ValueError(f"{path}:{line_number}: pose contains non-finite values")
            if np.linalg.norm(quaternion) == 0:
                raise ValueError(f"{path}:{line_number}: quaternion is zero")
            poses.append((name, position, quaternion))

    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


def sample_point_cloud(
    las_path: Path,
    ply_path: Path,
    target_num_points: int,
    voxel_size: float,
    random_ratio: float,
    seed: int,
) -> o3d.geometry.PointCloud:
    las = laspy.read(las_path)
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {las_path}")

    colors = None
    if all(hasattr(las, channel) for channel in ("red", "green", "blue")):
        colors = np.column_stack((las.red, las.green, las.blue)).astype(np.float64)
        colors /= 65535.0 if colors.max() > 255 else 255.0

    point_data = points if colors is None else np.column_stack((points, colors))

    if target_num_points > 0 and len(points) > target_num_points:
        point_data = two_stage_point_cloud_sample(
            point_data,
            target_num_points=target_num_points,
            voxel_size=voxel_size,
            random_ratio=random_ratio,
            rng=np.random.default_rng(seed),
        )
    points = point_data[:, :3]
    if colors is not None:
        colors = point_data[:, 3:]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    if not point_cloud.has_colors():
        point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size, 0.1), max_nn=30)
    )
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(ply_path), point_cloud, write_ascii=False):
        raise IOError(f"Failed to write point cloud: {ply_path}")
    print(f"[point cloud] {len(points)} -> {len(point_cloud.points)} points")
    print(f"[point cloud] wrote {ply_path}")
    return point_cloud


def points3d_dict(point_cloud: o3d.geometry.PointCloud) -> dict[int, Point3D]:
    xyz = np.asarray(point_cloud.points, dtype=np.float64)
    if point_cloud.has_colors():
        rgb = np.clip(np.asarray(point_cloud.colors) * 255, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    return {
        index + 1: Point3D(
            id=index + 1,
            xyz=point,
            rgb=rgb[index],
            error=0.0,
            image_ids=np.zeros(0, dtype=np.int32),
            point2D_idxs=np.zeros(0, dtype=np.int32),
        )
        for index, point in enumerate(xyz)
    }


def make_camera_models(cameras: dict[str, dict]) -> dict[str, Camera]:
    models = {}
    for camera_id, name in enumerate(("left", "right"), 1):
        if name not in cameras:
            continue
        camera = cameras[name]
        intrinsic = camera["intrinsic"]
        models[name] = Camera(
            id=camera_id,
            model="OPENCV",
            width=int(camera["width"]),
            height=int(camera["height"]),
            params=np.asarray(
                [
                    intrinsic["fl_x"],
                    intrinsic["fl_y"],
                    intrinsic["cx"],
                    intrinsic["cy"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
        )
    return models


def build_images(poses: list[tuple[str, np.ndarray, np.ndarray]], camera_models: dict[str, Camera]):
    images = {}
    for name, center, quaternion in poses:
        camera_name = name.split("/", 1)[0].lower()
        if camera_name not in camera_models:
            print(f"[warn] skip {name}: camera {camera_name!r} is not calibrated")
            continue
        rotation_cw = Rotation.from_quat(quaternion / np.linalg.norm(quaternion)).as_matrix()
        translation_cw = -rotation_cw @ center
        q_xyzw = Rotation.from_matrix(rotation_cw).as_quat()
        qvec = np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        image_id = len(images) + 1
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=translation_cw,
            camera_id=camera_models[camera_name].id,
            name=name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros(0, dtype=np.int64),
        )
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, help="Directory containing the three input files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output COLMAP project directory")
    parser.add_argument("--num-points", type=int, default=500000, help="Maximum output points; 0 keeps all")
    parser.add_argument("--voxel-size", type=float, default=0.5, help="Voxel size in point-cloud units; 0 disables voxel sampling")
    parser.add_argument("--random-ratio", type=float, default=0.6, help="Ratio of target points sampled randomly in the first stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the final point sampling")
    parser.add_argument("--format", choices=("bin", "txt", "both"), default="both", dest="model_format")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "lidar-post-sparse").resolve()
    cameras = load_calibration(input_dir / "calibration.json")
    poses = parse_pose_file(input_dir / "ImgPose.txt")
    camera_models = make_camera_models(cameras)
    images = build_images(poses, camera_models)
    sparse_dir = output_dir
    point_cloud = sample_point_cloud(
        input_dir / "colorized.las",
        sparse_dir / "points3D.ply",
        args.num_points,
        args.voxel_size,
        args.random_ratio,
        args.seed,
    )
    points = points3d_dict(point_cloud)
    camera_by_id = {camera.id: camera for camera in camera_models.values()}
    if args.model_format in {"bin", "both"}:
        write_model(camera_by_id, images, points, str(sparse_dir), ext=".bin")
    if args.model_format in {"txt", "both"}:
        write_model(camera_by_id, images, points, str(sparse_dir), ext=".txt")
    print(f"[done] cameras={len(camera_by_id)}, images={len(images)}, points={len(points)}")
    print(f"[done] COLMAP model: {sparse_dir}")


if __name__ == "__main__":
    main()