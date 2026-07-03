import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_image_list(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    images = []

    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in exts:
                images.append(os.path.join(root, f))

    images.sort()
    return images


def main():

    parser = argparse.ArgumentParser(
        description="Undistort images (COLMAP OPENCV)"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image folder"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder"
    )

    args = parser.parse_args()

    ##########################################################
    # Camera Parameters (Replace with your own)
    ##########################################################

    width = 2640
    height = 1978

    fx = 1858.3663607435581
    fy = 1858.4558171291492

    cx = 1331.7724485725901
    cy = 978.8354408256587

    k1 = -0.0022464254192097778
    k2 = 0.019165554165893023
    p1 = -0.0006329107261463302
    p2 = 0.0005926413702848853

    ##########################################################

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    dist = np.array(
        [k1, k2, p1, p2],
        dtype=np.float64
    )

    ##########################################################
    # New Camera Matrix
    ##########################################################

    new_K = K.copy()

    # 主点移动到图像中心
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0

    ##########################################################

    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        new_K,
        (width, height),
        cv2.CV_16SC2
    )

    ##########################################################

    print("=" * 70)

    print("Original Camera Matrix")
    print(K)

    print("\nDistortion Coefficients")
    print(dist)

    print("\nNew Camera Matrix")
    print(new_K)

    print("=" * 70)

    print("\nNew Intrinsics")

    print(f"fx = {new_K[0,0]:.12f}")
    print(f"fy = {new_K[1,1]:.12f}")
    print(f"cx = {new_K[0,2]:.12f}")
    print(f"cy = {new_K[1,2]:.12f}")

    print("\nCOLMAP Camera")

    print(
        f"1 PINHOLE {width} {height} "
        f"{new_K[0,0]:.12f} "
        f"{new_K[1,1]:.12f} "
        f"{new_K[0,2]:.12f} "
        f"{new_K[1,2]:.12f}"
    )

    ##########################################################

    os.makedirs(args.output, exist_ok=True)

    images = get_image_list(args.input)

    print(f"\nFound {len(images)} images.")

    for image_path in tqdm(images):

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print("Cannot read:", image_path)
            continue

        if img.shape[1] != width or img.shape[0] != height:
            print(f"Skip (resolution mismatch): {image_path}")
            continue

        undist = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        rel = os.path.relpath(image_path, args.input)

        save_path = os.path.join(args.output, rel)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cv2.imwrite(save_path, undist)

    print("\nDone.")


if __name__ == "__main__":
    main()