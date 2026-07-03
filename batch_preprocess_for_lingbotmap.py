import os
import sys
from PIL import Image
from utils import count_images_in_dir,  find_directories_by_name
import re
from typing import List
from natsort import natsorted

from argparse import ArgumentParser

def sort_filenames_by_prefix_number(filenames: List[str], prefix: str = "frame_") -> List[str]:
    """
    根据指定前缀后的数字对文件名列表进行数值排序（非字典序）
    
    :param filenames: 原始文件名列表，如 ["frame_10.jpg", "frame_1.jpg"]
    :param prefix: 指定的文件名前缀，如 "frame_"、"img_"、"shot_" 等
    :return: 按数字升序排序后的新列表
    """
    # 转义前缀中的正则特殊字符，并预编译正则表达式提升性能
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")

    def _extract_number(name: str) -> int:
        # search 会在字符串中查找前缀+数字的组合，不强制要求前缀在开头
        match = pattern.search(name)
        # 匹配成功返回整数，失败返回无穷大（自动排到末尾）
        return int(match.group(1)) if match else float('inf')

    return sorted(filenames, key=_extract_number)

def rename_files_with_nature_sort(files_path, img_dir):
    for idx, filename in enumerate(files_path, start=0):
        # old_path = os.path.join(img_dir, filename)
        old_path = filename
        # img_dir = os.path.dirname(old_path)
        name, ext = os.path.splitext(filename)
        # 格式化：frame_00001 五位补零
        new_name = f"{idx:06d}{ext}"
        new_path = os.path.join(img_dir, new_name)
        
        os.rename(old_path, new_path)
        print(f"{filename}  ->  {new_path}")

if __name__ == '__main__':

    argpaser = ArgumentParser(description="batch process input images for lingbot-map")
    # argpaser.add_argument("--input_dir", type=str, required=True)
    argpaser.add_argument("--dataset_root", "-ds", type=str, default="dataset", help='数据集根目录')

    args = argpaser.parse_args()


    dataset_path = find_directories_by_name(args.dataset_root, "input")
    if not dataset_path:
        print("没有找到任何数据集，程序退出。")
        sys.exit(0)

    dataset_path = sorted(dataset_path)

    for dataset_dir in dataset_path:
        print(f"current data dir: {dataset_dir}")
        # img_dir = os.path.join(dataset_dir, "input")
        img_dir = dataset_dir
        project_dir = os.path.dirname(img_dir)

        # 常见图片后缀，可自行加
        suffix_list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

        # 1. 获取所有图片文件，按原文件名排序
        img_num, _, files = count_images_in_dir(img_dir)
        files = natsorted(files)

        # 2. 批量重命名
        rename_files_with_nature_sort(files, img_dir)
        print("✅ 全部重命名完成")

        # 3. 裁切到 518 * 294 的倍数
        _, _, files = count_images_in_dir(img_dir)
        for file in files:
            img = Image.open(file)
            ori_width, ori_height = img.size
            if ori_width / ori_height > 518 / 294:
                target_height = ori_height
                target_width = round(target_height / 294 * 518)
                left = (ori_width - target_width) / 2
                top = 0
                right = left + target_width
                bottom = target_height 
            else:
                target_width = ori_width
                target_height = round(target_width / 518 * 294)
                left = 0
                right = target_width
                top = (ori_height - target_height) / 2
                bottom = top + target_height  
            print(f"original size: ({ori_width}, {ori_height}), target size: ({target_width}, {target_height})")
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(file)

        print("✅ 裁剪完成，已覆盖原图片！")    