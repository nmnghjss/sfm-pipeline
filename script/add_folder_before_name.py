import os
import sys
from argparse import ArgumentParser

# 常见图像扩展名（可根据需要增减）
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def rename_images(root_dir):
    """递归遍历 root_dir，为每个图像文件添加父文件夹名作为前缀"""
    if not os.path.isdir(root_dir):
        print(f"错误：'{root_dir}' 不是有效的文件夹路径")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 当前文件夹的名称（直接父文件夹）
        parent_folder = os.path.basename(dirpath)
        
        for filename in filenames:
            # 检查是否为图像文件
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            old_path = os.path.join(dirpath, filename)
            new_name = f"{parent_folder}_{filename}"
            new_path = os.path.join(dirpath, new_name)

            # 如果新文件名已存在，跳过并提示
            if os.path.exists(new_path):
                print(f"警告：跳过 '{old_path}'，因为 '{new_path}' 已存在")
                continue

            try:
                os.rename(old_path, new_path)
                print(f"已重命名：{old_path} -> {new_path}")
            except Exception as e:
                print(f"错误：无法重命名 '{old_path}'，原因：{e}")

if __name__ == "__main__":
    # 从命令行参数获取文件夹路径，若无则交互输入
    arg_parser = ArgumentParser(description="为图像文件添加父文件夹名作为前缀")
    arg_parser.add_argument("--folder", nargs="?", help="要处理的文件夹路径")
    args = arg_parser.parse_args()
    folder = args.folder

    rename_images(folder)