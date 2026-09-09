import os
from pathlib import Path

import piexif
from PIL import Image


def to_rational(value):
    """
    将浮点数转换成EXIF需要的(num, den)格式
    例如 4.5 -> (4500,1000)
    """
    return (int(round(value * 1000)), 1000)


def modify_exif(image_path, focal_length, focal35):
    """
    修改图像焦距信息

    Parameters
    ----------
    image_path : str
    focal_length : float
        实际焦距(mm)
    focal35 : int
        35mm等效焦距(mm)
    """

    img = Image.open(image_path)

    if "exif" in img.info:
        exif_dict = piexif.load(img.info["exif"])
    else:
        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "1st": {},
            "Interop": {},
            "thumbnail": None,
        }

    # 实际焦距
    exif_dict["Exif"][piexif.ExifIFD.FocalLength] = to_rational(focal_length)

    # 35mm等效焦距
    exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(focal35)

    exif_bytes = piexif.dump(exif_dict)

    img.save(image_path, exif=exif_bytes)
    print(f"Updated: {image_path}")


def process_folder(folder, focal_length, focal35):
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    folder = Path(folder)

    for file in folder.rglob("*"):
        if file.suffix in exts:
            try:
                modify_exif(str(file), focal_length, focal35)
            except Exception as e:
                print(f"Failed: {file}")
                print(e)


if __name__ == "__main__":

    # 图像目录
    image_folder = r"E:\M3D_Test_Data\town\input\B\10080427"

    # 修改为你的真实焦距(mm)
    focal_length = 0

    # 修改为35mm等效焦距(mm)
    focal35 = 0

    process_folder(image_folder, focal_length, focal35)