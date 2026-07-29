"""
Visualize COLMAP sparse reconstruction: camera poses and 3D point cloud.

Usage:
    python visualize_colmap.py -i /path/to/sparse/0
    python visualize_colmap.py -i /path/to/sparse/0 --filter_min_error 5.0 --max_points 50000
    python visualize_colmap.py -i /path/to/sparse --camera_scale 0.3 --point_size 1.5
"""

import os
import sys
import numpy as np
import open3d as o3d
from argparse import ArgumentParser

# Import COLMAP model reader from the same directory
from read_write_model import read_model, qvec2rotmat


def create_camera_frustum(R, t, scale=0.15, color=None):
    """
    Create an Open3D LineSet representing a camera frustum.

    Parameters
    ----------
    R : np.ndarray (3, 3)
        Rotation matrix (world-from-camera).
    t : np.ndarray (3,)
        Translation vector (camera center in world).
    scale : float
        Size of the frustum.
    color : list of float, optional
        RGB color for the frustum lines, default red.

    Returns
    -------
    o3d.geometry.LineSet
    """
    if color is None:
        color = [1.0, 0.0, 0.0]

    # Frustum points in camera coordinate system
    points = np.array(
        [
            [0, 0, 0],        # camera center
            [-1, -1, 2],      # bottom-left
            [1, -1, 2],       # bottom-right
            [1, 1, 2],        # top-right
            [-1, 1, 2],       # top-left
        ],
        dtype=np.float64,
    )
    points *= scale

    # Transform to world coordinates
    points_world = (R @ points.T).T + t

    # Lines connecting frustum points
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # near-plane edges
    ]

    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def main():
    parser = ArgumentParser(
        description="Visualize COLMAP sparse reconstruction (cameras + point cloud)"
    )
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="Path to COLMAP sparse model folder (e.g. sparse/0).",
    )
    parser.add_argument(
        "--camera_scale", type=float, default=0.15,
        help="Size of camera frustums (default: 0.15).",
    )
    parser.add_argument(
        "--point_size", type=float, default=1.0,
        help="Render point size for the point cloud (default: 1.0).",
    )
    parser.add_argument(
        "--max_points", type=int, default=None,
        help="Maximum number of 3D points to display (subsampled randomly).",
    )
    parser.add_argument(
        "--filter_min_error", type=float, default=None,
        help="Only show points whose reprojection error is below this threshold.",
    )
    parser.add_argument(
        "--window_name", type=str, default="COLMAP Reconstruction",
        help="Title of the visualization window.",
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Window width (default: 1920).",
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Window height (default: 1080).",
    )
    parser.add_argument(
        "--hide_cameras", action="store_true",
        help="Do not show camera frustums.",
    )
    parser.add_argument(
        "--hide_trajectory", action="store_true",
        help="Do not show the camera trajectory line.",
    )
    parser.add_argument(
        "--hide_points", action="store_true",
        help="Do not show the point cloud.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.isdir(input_path):
        print(f"Error: input path does not exist or is not a directory: {input_path}")
        sys.exit(1)

    # ---- load COLMAP model ----
    cameras, images, points3D = read_model(input_path)

    if images is None:
        print("Error: failed to read images from the model.")
        sys.exit(1)

    print(f"Loaded {len(images)} images (cameras).")
    if points3D is not None:
        print(f"Loaded {len(points3D)} 3D points.")
    else:
        print("No 3D points found.")

    # ---- build geometries ----
    geometries = []

    # Coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(axis)

    # Cameras (frustums)
    if not args.hide_cameras:
        image_ids_sorted = sorted(images.keys())
        num_images = len(image_ids_sorted)
        for idx, image_id in enumerate(image_ids_sorted):
            img = images[image_id]
            R = qvec2rotmat(img.qvec)
            tvec = img.tvec

            # Gradient color from blue (start) to red (end)
            frac = idx / max(num_images - 1, 1)
            color = [frac, 0.2, 1.0 - frac]

            frustum = create_camera_frustum(R, tvec, scale=args.camera_scale, color=color)
            geometries.append(frustum)

    # Trajectory line
    if not args.hide_trajectory:
        trajectory_centers = []
        for image_id in sorted(images.keys()):
            trajectory_centers.append(images[image_id].tvec)

        if len(trajectory_centers) > 1:
            traj_pts = np.array(trajectory_centers)
            traj = o3d.geometry.LineSet()
            traj.points = o3d.utility.Vector3dVector(traj_pts)
            lines = [[i, i + 1] for i in range(len(traj_pts) - 1)]
            traj.lines = o3d.utility.Vector2iVector(lines)
            traj.colors = o3d.utility.Vector3dVector(
                [[0.0, 1.0, 0.0] for _ in lines]
            )
            geometries.append(traj)

    # Point cloud
    if not args.hide_points and points3D is not None and len(points3D) > 0:
        pts = []
        cols = []
        for pt_id, pt in points3D.items():
            if args.filter_min_error is not None and pt.error > args.filter_min_error:
                continue
            pts.append(pt.xyz)
            cols.append(pt.rgb / 255.0)

        pts = np.array(pts)
        cols = np.array(cols)

        if len(pts) > 0:
            if args.max_points is not None and len(pts) > args.max_points:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(pts), args.max_points, replace=False)
                pts = pts[indices]
                cols = cols[indices]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            geometries.append(pcd)

    # ---- visualize ----
    print(f"Displaying {len(geometries)} geometry objects...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=args.window_name,
        width=args.width,
        height=args.height,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
