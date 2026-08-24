"""
将 GPS txt 文件转换为 COLMAP 风格先验位置 JSON 文件。

txt 格式（每行一列图像，兼容空格或逗号分隔）:
    <图像名称> <经度> <纬度> <高度>
    A-DSC00002 116.774837070 39.637907320 74.650000000
    B_441421_2018031601_4889.JPG,116.245044,24.379834,234

输出 JSON 格式:
    {
      "cord_type": "wgs84",
      "images": [
        {
          "name": "rig1/camera1/000001.jpg",
          "x": 116.774837070, "y": 39.637907320, "z": 74.65,
          "cov_xx": 4.0, "cov_yy": 4.0, "cov_zz": 16.0
        }
      ]
    }
其中 x 为经度、y 为纬度、z 为高度。

用法:
    python gps_txt_to_json.py --txt gps.txt --image_dir ./images \
        --output gps.json --cov_xx 4.0 --cov_yy 4.0 --cov_zz 16.0
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="将 GPS txt 文件转换为 JSON 先验位置文件")
    parser.add_argument("--txt", required=True, help="输入的 GPS txt 文件路径")
    parser.add_argument("--image_dir", "-i", required=True, help="图像根目录（图像可能位于其多级子目录中）")
    parser.add_argument("--output", "-o", required=True, help="输出的 JSON 文件路径")
    parser.add_argument("--cord_type", default="wgs84", help="坐标系类型，默认 wgs84")
    parser.add_argument("--cov_xx", type=float, default=1.0, help="cov_xx 协方差值，默认 4.0")
    parser.add_argument("--cov_yy", type=float, default=1.0, help="cov_yy 协方差值，默认 4.0")
    parser.add_argument("--cov_zz", type=float, default=4.0, help="cov_zz 协方差值，默认 16.0")
    return parser.parse_args()


def read_lines(txt_path: Path):
    """读取 txt 文件，兼容 UTF-8 / GBK 编码。"""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(txt_path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("无法以 UTF-8/GBK 编码读取文件: {}".format(txt_path))


def build_name_to_relpath(image_dir: Path):
    """遍历 image_dir 下所有多级子目录，建立 图像名称 -> 相对路径 的映射。

    同时以文件名（含扩展名）和 stem（不含扩展名）作为键，
    以便 txt 中的名称（如 A-DSC00002）匹配磁盘上的文件（如 A-DSC00002.jpg）。
    """
    mapping = {}
    for root, _dirs, files in os.walk(image_dir):
        for f in files:
            rel = (Path(root) / f).relative_to(image_dir).as_posix()
            mapping[f] = rel            # 含扩展名的完整文件名
            mapping[Path(f).stem] = rel  # 不含扩展名的名称
    return mapping


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        print(f"错误: image_dir 不存在或不是目录: {image_dir}", file=sys.stderr)
        sys.exit(1)

    txt_path = Path(args.txt)
    if not txt_path.is_file():
        print(f"错误: txt 文件不存在: {txt_path}", file=sys.stderr)
        sys.exit(1)

    mapping = build_name_to_relpath(image_dir)
    print(f"在 {image_dir} 下共找到 {len(mapping)} 个名称键（含 stem）")

    images = []
    missing = []
    for line_no, line in enumerate(read_lines(txt_path), 1):
        line = line.strip()
        if not line:
            continue
        # 兼容逗号分隔（CSV）与空格分隔两种格式
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        if len(parts) < 4:
            print(f"警告: 第 {line_no} 行列数不足 4 列，已跳过: {line}", file=sys.stderr)
            continue
        name = parts[0]
        try:
            lon, lat, alt = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            print(f"警告: 第 {line_no} 行数值解析失败，已跳过: {line}", file=sys.stderr)
            continue

        rel = mapping.get(name)
        if rel is None:
            missing.append(name)
            continue
        images.append({
            "name": rel,
            "x": lon,
            "y": lat,
            "z": alt,
            "cov_xx": args.cov_xx,
            "cov_yy": args.cov_yy,
            "cov_zz": args.cov_zz,
        })

    if missing:
        preview = ", ".join(missing[:10])
        more = "" if len(missing) <= 10 else f" ... 等共 {len(missing)} 个"
        print(f"警告: 有 {len(missing)} 个图像名称未在目录中找到: {preview}{more}", file=sys.stderr)

    data = {"cord_type": args.cord_type, "images": images}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"成功: 共写入 {len(images)} 条记录到 {out_path}")


if __name__ == "__main__":
    main()
