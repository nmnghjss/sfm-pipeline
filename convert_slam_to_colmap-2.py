import os
import shutil
import numpy as np
import cv2
import open3d as o3d
from argparse import ArgumentParser

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    Return format:
        qw qx qy qz
    """

    q = np.empty(4, dtype=np.float64)

    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)

        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s

    else:

        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

            s = 2.0 * np.sqrt(
                1.0 + R[0, 0] - R[1, 1] - R[2, 2]
            )

            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s

        elif R[1, 1] > R[2, 2]:

            s = 2.0 * np.sqrt(
                1.0 + R[1, 1] - R[0, 0] - R[2, 2]
            )

            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s

        else:

            s = 2.0 * np.sqrt(
                1.0 + R[2, 2] - R[0, 0] - R[1, 1]
            )

            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    q /= np.linalg.norm(q)

    return q


def load_camera_intrinsics(slam_dir):

    intrinsics_txt = os.path.join(slam_dir, "intrinsics.txt")

    with open(intrinsics_txt, "r") as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip() and not line.strip().startswith("#")]

    fx, fy, cx, cy = map(float, lines[0].split())

    k1 = k2 = p1 = p2 = k3 = 0.0
    if len(lines) > 1:
        k1, k2, p1, p2, k3 = map(float, lines[1].split())

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3
    }


def load_slam_data(slam_dir):

    intrinsics = load_camera_intrinsics(slam_dir)

    poses_txt = os.path.join(slam_dir, "poses.txt")

    with open(poses_txt, "r") as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip()]

    ########################################################
    # poses
    ########################################################

    poses = []
    for line in lines[1:]:
        vals = list(map(float, line.split()))
        frame_id = int(vals[0])
        rvec = np.array(vals[1:4], dtype=np.float64)
        twc = np.array(vals[4:7], dtype=np.float64)
        poses.append((rvec, twc))

    ########################################################
    # image filenames
    ########################################################

    filename_txt = os.path.join(
        slam_dir,
        "keyframe_filenames.txt"
    )

    with open(filename_txt, "r") as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip()]

    image_names = []   
    for line in lines:
        vals = list(map(str, line.split()))
        img_id = int(vals[0])
        img_name = vals[1]
        image_names.append(img_name)


    assert len(image_names) == len(poses)
    num_keyframes = len(image_names)
    ########################################################
    # points3d
    ########################################################
    points3d_txt = os.path.join(
        slam_dir,
        "points3d.txt"
    )

    points3d = []
    with open(points3d_txt, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            xyz = list(map(float, line.split()))
            points3d.append(xyz)

    return {
        "num_keyframes": num_keyframes,
        "intrinsics": intrinsics,
        "poses": poses,
        "image_names": image_names,
        "points3d": points3d
    }

def get_image_size(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise RuntimeError(
            f"Cannot read image: {image_path}"
        )

    h, w = img.shape[:2]

    return w, h


def write_colmap_model(
        slam_dir,
        output_dir,
        slam_data):

    os.makedirs(output_dir, exist_ok=True)

    ########################################################
    # images folder
    ########################################################

    output_images = os.path.join(
        output_dir,
        "images"
    )

    os.makedirs(output_images, exist_ok=True)

    src_keyframes = os.path.join(
        slam_dir,
        "keyframes"
    )

    for name in slam_data["image_names"]:

        src = os.path.join(src_keyframes, name)
        dst = os.path.join(output_images, name)

        shutil.copy(src, dst)

    ########################################################
    # image size
    ########################################################

    first_img = os.path.join(
        output_images,
        slam_data["image_names"][0]
    )

    width, height = get_image_size(first_img)

    ########################################################
    # cameras.txt
    ########################################################

    cameras_out = os.path.join(
        output_dir,
        "cameras.txt"
    )

    with open(cameras_out, "w") as f:

        f.write(
            "# Camera list with one line of data per camera:\n"
        )

        f.write(
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        )

        f.write("# Number of cameras: 1\n")

        f.write(
            f"1 OPENCV "
            f"{width} {height} "
            f"{slam_data['intrinsics']['fx']} "
            f"{slam_data['intrinsics']['fy']} "
            f"{slam_data['intrinsics']['cx']} "
            f"{slam_data['intrinsics']['cy']} "
            f"{slam_data['intrinsics']['k1']} "
            f"{slam_data['intrinsics']['k2']} "
            f"{slam_data['intrinsics']['p1']} "
            f"{slam_data['intrinsics']['p2']}\n"
        )

    ########################################################
    # images.txt
    ########################################################

    images_out = os.path.join(
        output_dir,
        "images.txt"
    )

    with open(images_out, "w") as f:

        f.write(
            "# Image list with two lines of data per image:\n"
        )

        f.write(
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )

        f.write(
            "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
        )

        f.write(
            f"# Number of images: "
            f"{len(slam_data['poses'])}\n"
        )

        for idx, (
                pose,
                image_name) in enumerate(
                zip(
                    slam_data["poses"],
                    slam_data["image_names"])):

            rvec, twc = pose

            ################################################
            # Rodrigues -> Rwc
            ################################################

            Rwc, _ = cv2.Rodrigues(rvec)

            ################################################
            # Convert Twc -> Tcw
            ################################################

            # Rcw = Rwc.T
            # tcw = -Rcw @ twc

            Rcw = Rwc
            tcw = twc

            ################################################
            # quaternion
            ################################################

            q = rotation_matrix_to_quaternion(Rcw)
            qw, qx, qy, qz = q
            tx, ty, tz = tcw

            image_id = idx

            f.write(
                f"{image_id} "
                f"{qw} {qx} {qy} {qz} "
                f"{tx} {ty} {tz} "
                f"1 "
                f"{image_name}\n"
            )

            ################################################
            # empty POINTS2D line
            ################################################

            f.write("\n")

    ########################################################
    # points3D.txt
    ########################################################

    points_out = os.path.join(
        output_dir,
        "points3D.txt"
    )

    with open(points_out, "w") as f:

        f.write(
            "# 3D point list with one line of data per point:\n"
        )

        f.write(
            "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"
        )

        f.write(
            f"# Number of points: "
            f"{len(slam_data['points3d'])}\n"
        )

        for idx, xyz in enumerate(
                slam_data["points3d"]):

            point_id = idx + 1

            x, y, z = xyz

            ################################################
            # RGB 默认白色
            ################################################

            r, g, b = 128, 128, 128

            ################################################
            # error 默认0
            ################################################

            error = 0

            ################################################
            # TRACK 为空
            ################################################

            f.write(
                f"{point_id} "
                f"{x} {y} {z} "
                f"{r} {g} {b} "
                f"{error}\n"
            )

    print("====================================")
    print("COLMAP model export done.")
    print(f"Output: {output_dir}")
    print("====================================")



def create_camera_frustum(
        Rwc,
        twc,
        scale=0.15):

    points = np.array([
        [0, 0, 0],
        [-1, -1, 2],
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2]
    ], dtype=np.float64)

    points *= scale

    ########################################################
    # camera -> world
    ########################################################

    points_world = (Rwc @ points.T).T + twc

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1]
    ]

    colors = [[1, 0, 0] for _ in lines]

    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(
        points_world
    )

    line_set.lines = o3d.utility.Vector2iVector(
        lines
    )

    line_set.colors = o3d.utility.Vector3dVector(
        colors
    )

    return line_set


############################################################
# Visualization
############################################################

def visualize_slam(slam_data):

    geometries = []

    ########################################################
    # world coordinate
    ########################################################

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5
    )

    geometries.append(axis)

    ########################################################
    # point cloud
    ########################################################

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        slam_data["points3d"]
    )

    ########################################################
    # white
    ########################################################

    # colors = np.ones_like(slam_data["points3d"])
    colors = np.tile(np.array([[0, 1, 0]]), (len(slam_data["points3d"]), 1))

    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries.append(pcd)

    ########################################################
    # cameras
    ########################################################

    trajectory_points = []

    for rvec, tcw in slam_data["poses"]:

        Rcw, _ = cv2.Rodrigues(rvec)

        # ============ convert to world coordinate ============
        Rwc = Rcw.T
        twc = -Rcw.T @ tcw        

        ####################################################
        # frustum
        ####################################################

        cam = create_camera_frustum(
            Rwc,
            twc
        )

        geometries.append(cam)

        trajectory_points.append(twc)

    ########################################################
    # trajectory line
    ########################################################

    trajectory_points = np.array(trajectory_points)

    traj = o3d.geometry.LineSet()

    traj.points = o3d.utility.Vector3dVector(
        trajectory_points
    )

    lines = []

    for i in range(len(trajectory_points) - 1):

        lines.append([i, i + 1])

    traj.lines = o3d.utility.Vector2iVector(lines)

    ########################################################
    # green trajectory
    ########################################################

    traj.colors = o3d.utility.Vector3dVector(
        [[0, 1, 0] for _ in lines]
    )

    geometries.append(traj)

    ########################################################
    # show
    ########################################################

    o3d.visualization.draw_geometries(
        geometries,
        window_name="SLAM Visualization",
        width=1600,
        height=900
    )


if __name__ == "__main__":

    arg_parser = ArgumentParser(
        description="Convert SLAM data to COLMAP format"
    )
    arg_parser.add_argument("--input_dir", "-i", help="Directory containing SLAM data")
    arg_parser.add_argument("--output_dir", "-o", help="Output directory for COLMAP model")

    args = arg_parser.parse_args()

    slam_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    slam_data = load_slam_data(slam_dir)

    visualize_slam(slam_data)

    write_colmap_model(
        slam_dir,
        output_dir,
        slam_data
    )