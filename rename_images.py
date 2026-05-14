import os
from PIL import Image
from utils import count_images_in_dir
# ========= 在这里改你的图片文件夹路径 =========
img_dir = r"E:\\datas\\Ling-Bot-Map\\xfz\\input"
# ============================================

# 常见图片后缀，可自行加
suffix_list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

# 1. 获取所有图片文件，按原文件名排序
img_num, _, files = count_images_in_dir(img_dir)

# 按原文件名排序
files.sort()

# 2. 批量重命名
for idx, filename in enumerate(files, start=0):
    old_path = os.path.join(img_dir, filename)
    name, ext = os.path.splitext(filename)
    # 格式化：frame_00001 五位补零
    new_name = f"frame_{idx:06d}{ext}"
    new_path = os.path.join(img_dir, new_name)
    
    os.rename(old_path, new_path)
    print(f"{filename}  ->  {new_name}")

print("✅ 全部重命名完成")

# 3. 覆盖保存！直接覆盖原文件（关键）
left = 35
top = 0
right = 4557
bottom = 3056
_, _, files = count_images_in_dir(img_dir)
for file in files:
    img = Image.open(file)
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(file)

print("✅ 裁剪完成，已覆盖原图片！")    