
import os
from argparse import ArgumentParser
import sys
import numpy as np
# from get_depth_maps import get_depth_maps
# from correct_depth_maps import compute_scale_and_offset
# from train import *
from utils import find_directories_by_name, count_images_in_dir
from read_write_model import read_model
from measure_pose import align_and_compute_error, PoseError
import csv


def resource_path():
    try:
        base_path = sys._MEIPASS  # 打包后：指向临时解压目录
        print(print("sys._MEIPASS:", sys._MEIPASS))
        print("os.listdir(sys._MEIPASS):", os.listdir(sys._MEIPASS))
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))  # 开发时：当前文件所在目录
    return base_path


def get_project_root():
    """返回项目根目录（兼容开发和打包环境）"""
    if getattr(sys, 'frozen', False):
        # 打包模式：程序已被冻结（frozen）
        return os.path.dirname(sys.executable)
    else:
        # 开发模式：直接运行 .py 文件
        return os.path.dirname(os.path.abspath(__file__))


def save_metrics_to_csv(metrics_dict, output_path, fieldnames=None):
    """
    将 metrics 字典按键排序后保存为 CSV 文件。支持对象或字典值。
    
    Args:
        metrics_dict (dict): 键为 str，值为对象或字典。
        output_path (str): 保存路径。
        fieldnames (list): 自定义列顺序，如果为 None 则使用默认顺序。
    """
    if not metrics_dict:
        print("警告: 字典为空，取消保存。")
        return

    # 1. 确保目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 2. 获取表头 (从第一个对象动态获取所有属性名)
    first_obj = next(iter(metrics_dict.values()))
    # 兼容对象和字典两种形式
    if isinstance(first_obj, dict):
        first_obj_dict = first_obj
    else:
        first_obj_dict = vars(first_obj)
    
    if fieldnames is None:
        fieldnames = ['id'] + list(first_obj_dict.keys())
    else:
        # 确保 'id' 在第一列
        if 'id' not in fieldnames:
            fieldnames = ['id'] + fieldnames

    # try:
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 3. 按 key (即 id) 的字母顺序排序
        for key in sorted(metrics_dict.keys()):
            obj = metrics_dict[key]

            # 转换为字典并写入（兼容对象和字典）
            if isinstance(obj, dict):
                row = obj.copy()
            else:
                row = vars(obj).copy()
            
            row['id'] = key
            
            # 对浮点数进行格式化，只保留2位小数
            for key_name in row:
                if isinstance(row[key_name], float):
                    row[key_name] = f"{row[key_name]:.2f}"
            
            writer.writerow(row)
            
    print(f"成功按 ID 排序并保存至: {output_path}")

    # except Exception as e:
    #     print(f"保存失败: {e}")


if __name__ == '__main__':

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--dataset_root", "-ds",type=str, default="datasets", help='数据集根目录')
    parser.add_argument("--base_dir", "-b", type=str, required=True, help='3DGS模型输出目录')
    parser.add_argument("--update_dir", "-u", type=str, required=True, help="update sfm output dir name")
    parser.add_argument("--output_file_name", "-o",type=str, default="sfm_error.csv", help='保存所有数据集时间和指标的csv文件名')
    parser.add_argument("--visualize", action='store_true', default=False, help='是否可视化误差分布图')

    args = parser.parse_args()
    # args.save_iterations.append(args.iterations)

    dataset_path = find_directories_by_name(args.dataset_root, "input")
    if not dataset_path:
        print("没有找到任何数据集，程序退出。")
        sys.exit(0)

    dataset_path = sorted(dataset_path)

    # =========================处理每个数据集===================================================================================
    pose_erroe_list = {}

    for dataset_dir in dataset_path:
        data_path = os.path.dirname(dataset_dir)        
        print(f"Processing dataset at: {data_path}")

        input_images_num = count_images_in_dir(os.path.join(data_path, "input"))
        print(f"    输入图像数量: {input_images_num}")

        base_sfm_dir = os.path.join(data_path, args.base_dir, "sparse", "0")
        update_sfm_dir = os.path.join(data_path, args.update_dir, "sparse", "0")
        base_registered_num = count_images_in_dir(os.path.join(data_path, args.base_dir, "images"))
        update_registered_num = count_images_in_dir(os.path.join(data_path, args.update_dir, "images"))
        print(f"    基线 SfM 注册图像数量: {base_registered_num}")
        print(f"    更新 SfM 注册图像数量: {update_registered_num}")

        base_cameras, base_images, base_points3D = read_model(base_sfm_dir, ext=".bin")
        update_cameras, update_images, update_points3D = read_model(update_sfm_dir, ext=".bin")

        if base_images is None or update_images is None:
            print("error: base image or update image is None, skipping this dataset.")
            errors = PoseError()
            errors.ate_error_mean = 10000
            errors.rotate_angle_error_mean = 10000
        else:
            errors = align_and_compute_error(base_images, update_images, visualize=args.visualize)
            errors.base_registered_num = base_registered_num
            errors.update_registered_num = update_registered_num
            errors.base_registered_ratio = base_registered_num / input_images_num if input_images_num > 0 else 0.0
            errors.update_registered_ratio = update_registered_num / input_images_num if input_images_num > 0 else 0.0
            errors.registered_diff = update_registered_num - base_registered_num
            errors.registered_ratio_diff = errors.update_registered_ratio - errors.base_registered_ratio

        if errors is not None:
            pose_erroe_list[data_path] = errors

            print(f"Dataset: {data_path}")
            print(f"    相机中心位置误差统计信息: 均值={errors.ate_error_mean:.3f} 米, 标准差={errors.ate_error_std:.3f} 米, rmse = {errors.ate_error_rmse:.3f} 米")
            print(f"    相机旋转误差统计信息: 均值={errors.rotate_angle_error_mean:.3f} 度, 标准差={errors.rotate_angle_error_std:.3f} 度, rmse = {errors.rotate_angle_error_rmse:.3f} 度, median = {errors.rotate_angle_error_median:.3f} 度")
        else:
            print(f"Dataset: {data_path} - No valid matches found for error computation.")


    # =========================总结所有数据集的误差===========================================================================
    print("\n=== Summary of All Datasets ===")
    # 计算均值
    if errors:
        avg_pose_error = PoseError()
        avg_pose_error.ate_error_mean = np.mean([e.ate_error_mean for e in pose_erroe_list.values()])
        avg_pose_error.ate_error_std = np.mean([e.ate_error_std for e in pose_erroe_list.values()])
        avg_pose_error.ate_error_rmse = np.mean([e.ate_error_rmse for e in pose_erroe_list.values()])
        avg_pose_error.ate_error_median = np.mean([e.ate_error_median for e in pose_erroe_list.values()])
        avg_pose_error.ate_error_max = np.mean([e.ate_error_max for e in pose_erroe_list.values()])
        avg_pose_error.ate_error_p90 = np.mean([e.ate_error_p90 for e in pose_erroe_list.values()])
        avg_pose_error.rotate_angle_error_mean = np.mean([e.rotate_angle_error_mean for e in pose_erroe_list.values()])
        avg_pose_error.rotate_angle_error_std = np.mean([e.rotate_angle_error_std for e in pose_erroe_list.values()])
        avg_pose_error.rotate_angle_error_rmse = np.mean([e.rotate_angle_error_rmse for e in pose_erroe_list.values()])
        avg_pose_error.rotate_angle_error_median = np.mean([e.rotate_angle_error_median for e in pose_erroe_list.values()]) 
        avg_pose_error.rotate_angle_error_max = np.mean([e.rotate_angle_error_max for e in pose_erroe_list.values()])
        avg_pose_error.rotate_angle_error_p90 = np.mean([e.rotate_angle_error_p90 for e in pose_erroe_list.values()])
        # 打印均值信息
        print("\n平均值统计:")
        print("-" * 60)
        print(f"平均相机中心位置误差: 均值={avg_pose_error.ate_error_mean:.3f} 米, 标准差={avg_pose_error.ate_error_std:.3f} 米, rmse = {avg_pose_error.ate_error_rmse:.3f} 米")
        print(f"平均相机旋转误差: 均值={avg_pose_error.rotate_angle_error_mean:.3f} 度, 标准差={avg_pose_error.rotate_angle_error_std:.3f} 度, rmse = {avg_pose_error.rotate_angle_error_rmse:.3f} 度, median = {avg_pose_error.rotate_angle_error_median:.3f} 度")
    else:
        print("没有有效的误差数据可供计算平均值。")
 
    custom_fieldnames = [
        'base_registered_num', 'update_registered_num', 'base_registered_ratio', 'update_registered_ratio', 'registered_ratio_diff',
        'ate_error_mean', 'ate_error_median', 'ate_error_std', 'ate_error_rmse', 'ate_error_p90', 'ate_error_max',
        'rotate_angle_error_mean', 'rotate_angle_error_median', 'rotate_angle_error_std', 'rotate_angle_error_rmse', "rotate_angle_error_p90", 'rotate_angle_error_max',
        'registered_diff'
    ]
    
    save_metrics_to_csv(pose_erroe_list, os.path.join(args.dataset_root, args.output_file_name), fieldnames=custom_fieldnames)