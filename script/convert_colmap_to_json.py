"""
将 COLMAP 稀疏重建结果转换为 COLMAP 风格先验位置 JSON 文件。

读取 COLMAP 稀疏重建文件夹（cameras / images / points3D，.bin 或 .txt），
根据每张图像的位姿（四元数 qvec + 平移 tvec）计算其相机中心在世界坐标系中的
坐标（COLMAP 约定：C = -R^T · t），并输出为 JSON。

输出 JSON 格式（与 convert_gps_txt_to_json.py 一致）:
    {
      "cord_type": "colmap",
      "images": [
        {
          "name": "rig1/camera1/000001.jpg",
          "x": 1.234, "y": 5.678, "z": 3.210,
          "cov_xx": 1.0, "cov_yy": 1.0, "cov_zz": 4.0
        }
      ]
    }
其中 x/y/z 为相机中心在 COLMAP 世界坐标系下的三维坐标，name 为模型中的
图像相对路径（可直接用于 COLMAP 的 pose_prior_importer / 图像对匹配）。

用法:
    python convert_colmap_to_json.py --sparse ./sparse/0 --output prior.json \
        [--cord_type colmap] [--cov_xx 1.0 --cov_yy 1.0 --cov_zz 4.0]
"""

import argparse
import json
import sys
from pathlib import Path

# 允许从项目根目录导入 read_write_model 等模块
parent_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(parent_dir))

from read_write_model import read_model, qvec2rotmat  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="将 COLMAP 稀疏重建结果转换为 JSON 先验位置文件")
    parser.add_argument(
        "--sparse", "-s", required=True,
        help="COLMAP sparse 重建结果文件夹（含 cameras/images/points3D 的 .bin 或 .txt）",
    )
    parser.add_argument("--output", "-o", required=True, help="输出的 JSON 文件路径")
    parser.add_argument(
        "--cord_type", default="cartesian",
        help="坐标系类型，默认 colmap（COLMAP 局部世界坐标系）",
    )
    parser.add_argument("--cov_xx", type=float, default=1.0, help="cov_xx 协方差值")
    parser.add_argument("--cov_yy", type=float, default=1.0, help="cov_yy 协方差值")
    parser.add_argument("--cov_zz", type=float, default=1.0, help="cov_zz 协方差值")
    return parser.parse_args()


def compute_camera_center(img):
    """计算图像在世界坐标系中的相机中心：C = -R^T · t（COLMAP 约定）

    参数：
        img: read_write_model.Image（含 qvec、tvec）

    返回：
        np.ndarray, shape (3,)
    """
    R = qvec2rotmat(img.qvec)
    return -R.T @ img.tvec


def convert_colmap_to_prior_json(
    sparse_dir,
    json_path,
    cord_type="cartesian",
    cov_xx=1.0,
    cov_yy=1.0,
    cov_zz=1.0,
):
    """将 COLMAP 稀疏重建结果转换为先验位置 JSON 文件。

    参数：
        sparse_dir: COLMAP 稀疏重建目录，包含 cameras、images、points3D
            的 .bin 或 .txt 文件。
        json_path: 输出先验位置 JSON 文件路径。
        cord_type: 输出 JSON 中的坐标系类型字段。
        cov_xx: X 方向协方差值。
        cov_yy: Y 方向协方差值。
        cov_zz: Z 方向协方差值。

    返回：
        写入 JSON 的图像位姿记录数量。

    异常：
        FileNotFoundError: sparse_dir 不存在或不是目录。
        ValueError: sparse_dir 中没有有效的 COLMAP 稀疏模型。
    """
    sparse_dir = Path(sparse_dir).resolve()
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"sparse 目录不存在或不是目录: {sparse_dir}")

    # 读取 COLMAP 稀疏模型（自动检测 .bin / .txt）
    cameras, images, points3D = read_model(str(sparse_dir))
    if images is None:
        raise ValueError(
            f"无法在 {sparse_dir} 中检测到有效的 COLMAP 模型"
            "（需要 cameras/images/points3D 的 .bin 或 .txt）"
        )

    print(
        f"读取到 {len(images)} 张图像, {len(cameras)} 个相机, "
        f"{len(points3D)} 个三维点"
    )

    records = []
    for img_id in sorted(images.keys()):
        img = images[img_id]
        C = compute_camera_center(img)
        records.append({
            "name": img.name,
            "x": float(C[0]),
            "y": float(C[1]),
            "z": float(C[2]),
            "cov_xx": cov_xx,
            "cov_yy": cov_yy,
            "cov_zz": cov_zz,
        })

    data = {"cord_type": cord_type, "images": records}

    out_path = Path(json_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"成功: 共写入 {len(records)} 条记录到 {out_path}")
    return len(records)


def main():
    args = parse_args()
    try:
        convert_colmap_to_prior_json(
            sparse_dir=args.sparse,
            json_path=args.output,
            cord_type=args.cord_type,
            cov_xx=args.cov_xx,
            cov_yy=args.cov_yy,
            cov_zz=args.cov_zz,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
