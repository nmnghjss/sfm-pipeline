import os
import sys
from pathlib import Path
from PIL import Image
import piexif
import cv2
import numpy as np

# 当前文件所在目录的父目录
parent_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(parent_dir))

from utils import count_images_in_dir

# ========= 在这里改你的图片文件夹路径 =========
img_dir = r"E:\\images2400\\input"
output_dir = r"E:\\images2400\\resized1600-exif\\input"
os.makedirs(output_dir, exist_ok=True)
# ============================================

# 常见图片后缀，可自行加
suffix_list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# 1. 获取所有图片文件，按原文件名排序
img_num, _, files = count_images_in_dir(img_dir)

# 按原文件名排序
files.sort()
dst_width = 1600
# 2. 批量resize
success_count = 0
failed_count = 0
for idx, img_path in enumerate(files, start=0):    
    try:
        # 使用OpenCV读取图像（比PIL快）
        img = cv2.imread(img_path)
        if img is None:
            print(f"error: {os.path.basename(img_path)}  - 无法读取图像（可能文件损坏）")
            failed_count += 1
            continue
        
        height, width = img.shape[:2]
        dst_height = round(height * dst_width / width)
        # OpenCV的resize性能更好
        img_resized = cv2.resize(img, (dst_width, dst_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 保存图像
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img_resized)
        
        # 保留EXIF信息（仅对JPEG有效）
        if img_path.lower().endswith(('.jpg', '.jpeg')):
            try:
                exif_dict = piexif.load(img_path)
                exif_data = piexif.dump(exif_dict)
                # 用PIL重新打开并保存EXIF信息
                img_pil = Image.open(output_path)
                img_pil.save(output_path, exif=exif_data)
            except:
                pass  # 如果没有EXIF信息或保存失败，直接跳过
        
        print(f"{os.path.basename(img_path)}  ->  {os.path.basename(img_path)}")
        success_count += 1
        
    except Exception as e:
        print(f"error: {os.path.basename(img_path)}  - 处理失败: {str(e)}")
        failed_count += 1
        continue

print(f"\n✅ resize完成 - 成功: {success_count}, 失败: {failed_count}")

