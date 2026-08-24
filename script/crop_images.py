"""
裁剪图像脚本：以图像中心为中心，裁取指定宽高的区域，保存到输出文件夹。

用法:
    python crop_images.py -i /path/to/input -o /path/to/output -w 512 -h 512
    python crop_images.py -i input -o output --crop_width 640 --crop_height 480
"""

import argparse
import os
from pathlib import Path

import cv2


def find_images(root_dir, extensions=None):
    """递归查找所有图像文件，返回 Path 列表（已排序）。"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    images = []
    root = Path(root_dir)
    for file in root.rglob('*'):
        if file.is_file() and file.suffix.lower() in extensions:
            images.append(file)
    return sorted(images)


def crop_center(image, crop_width, crop_height):
    """以图像中心为基准裁取 crop_width x crop_height 区域，越界时自动收拢到图像边界。"""
    h, w = image.shape[:2]

    # 裁取尺寸不能超过原图
    cw = min(crop_width, w)
    ch = min(crop_height, h)

    # 以中心对齐计算左上角坐标（cw/ch 不超过原图，因此必然落在合法范围内）
    x = (w - cw) // 2
    y = (h - ch) // 2

    return image[y:y + ch, x:x + cw]


def main():
    parser = argparse.ArgumentParser(description="以图像中心为中心裁剪图像")
    parser.add_argument("--input_dir", "-i", required=True, help="输入图像文件夹")
    parser.add_argument("--output_dir", "-o", required=True, help="输出文件夹")
    parser.add_argument("--crop_width", "-cw", type=int, default=512, help="裁剪区域宽度（像素）")
    parser.add_argument("--crop_height", "-ch", type=int, default=512, help="裁剪区域高度（像素）")
    parser.add_argument("--extensions", type=str, default="",
                        help="可选：逗号分隔的图像扩展名，如 '.jpg,.png'（默认常见格式）")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.crop_width <= 0 or args.crop_height <= 0:
        print("Error: crop_width and crop_height must be positive integers.")
        return

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        return

    extensions = None
    if args.extensions.strip():
        extensions = {
            ext.strip().lower() if ext.strip().startswith('.') else '.' + ext.strip().lower()
            for ext in args.extensions.split(',') if ext.strip()
        }

    images = find_images(input_dir, extensions)
    if not images:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(images)} images")
    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    failed = 0
    for img_path in images:
        # 保持输入目录下的相对子目录结构
        rel_path = img_path.relative_to(input_dir)
        out_path = Path(output_dir) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read image: {img_path}")
            failed += 1
            continue

        cropped = crop_center(img, args.crop_width, args.crop_height)

        if not cv2.imwrite(str(out_path), cropped):
            print(f"  [WARN] Failed to write: {out_path}")
            failed += 1
            continue
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(images)}")

    print(f"Done. Cropped {processed} images, failed {failed}. Output: {output_dir}")


if __name__ == "__main__":
    main()
