from pathlib import Path
from exif import Image
from argparse import ArgumentParser


def remove_orientation(image_path):
    """
    删除 JPEG EXIF 中的 Orientation。
    
    注意：
    1. 不旋转图像
    2. 不解码 JPEG 像素
    3. 不重新编码 JPEG
    4. 只修改 EXIF 元数据
    """
    with open(image_path, "rb") as f:
        image = Image(f.read())

    # 获取 Orientation
    orientation = image.get("orientation")

    if orientation is None:
        return False, "没有 Orientation"

    # 删除 Orientation
    image.delete("orientation")

    # 写回文件
    with open(image_path, "wb") as f:
        f.write(image.get_file())

    return True, f"原 Orientation = {orientation}"


def process_folder(folder):
    folder = Path(folder)

    extensions = {
        ".jpg",
        ".jpeg",
        ".JPG",
        ".JPEG",
    }

    total = 0
    processed = 0
    skipped = 0
    failed = 0

    for image_path in folder.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix not in extensions:
            continue

        total += 1

        try:
            success, message = remove_orientation(image_path)

            if success:
                processed += 1
                print(f"[已处理] {image_path.name}: {message}")
            else:
                skipped += 1
                print(f"[跳过]   {image_path.name}: {message}")

        except Exception as e:
            failed += 1
            print(f"[失败]   {image_path.name}: {e}")

    print()
    print("=" * 60)
    print(f"总图片数: {total}")
    print(f"已处理:   {processed}")
    print(f"已跳过:   {skipped}")
    print(f"失败:     {failed}")
    print("=" * 60)


if __name__ == "__main__":

    parser = ArgumentParser(description="Remove EXIF Orientation from JPEG images in a folder")
    parser.add_argument("--folder", "-f", required=True, help="图像文件夹路径")
    args = parser.parse_args()

    process_folder(args.folder)