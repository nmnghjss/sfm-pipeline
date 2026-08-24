import os
import shutil
import sys
from argparse import ArgumentParser

# 支持的图像扩展名
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".gif"
}


def split_images(folder):
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        print(f"错误：文件夹不存在：{folder}")
        return

    # 创建目标文件夹
    even_folder = os.path.join(folder, "even")
    odd_folder = os.path.join(folder, "odd")

    os.makedirs(even_folder, exist_ok=True)
    os.makedirs(odd_folder, exist_ok=True)

    even_count = 0
    odd_count = 0
    skipped_count = 0

    # 只遍历当前文件夹
    for filename in os.listdir(folder):

        src_path = os.path.join(folder, filename)

        # 跳过文件夹
        if not os.path.isfile(src_path):
            continue

        # 判断是否为图像
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        # 获取不带扩展名的文件名
        stem = os.path.splitext(filename)[0]

        # 文件名最后一位必须是数字
        if not stem or not stem[-1].isdigit():
            print(f"跳过（文件名最后一位不是数字）：{filename}")
            skipped_count += 1
            continue

        # 判断奇偶
        last_digit = int(stem[-1])

        if last_digit % 2 == 0:
            dst_folder = even_folder
            even_count += 1
        else:
            dst_folder = odd_folder
            odd_count += 1

        dst_path = os.path.join(dst_folder, filename)

        # 如果目标文件已经存在，避免直接覆盖
        if os.path.exists(dst_path):
            print(f"跳过（目标文件已存在）：{filename}")
            continue

        shutil.move(src_path, dst_path)
        print(f"{filename} -> {os.path.basename(dst_folder)}/")

    print("\n处理完成")
    print(f"偶数图像：{even_count}")
    print(f"奇数图像：{odd_count}")
    print(f"跳过图像：{skipped_count}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Split images into even and odd folders based on filename")
    parser.add_argument("--folder", help="Image folder path")
    args = parser.parse_args()

    split_images(args.folder)
