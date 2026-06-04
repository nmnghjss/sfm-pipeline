

from argparse import ArgumentParser, Namespace
import shutil
import os, time
import logging
import shutil
import sys

# from get_depth_maps import get_depth_maps
# from correct_depth_maps import compute_scale_and_offset
# from train import *
from utils import count_image_files, copy_file, delete_directory, delete_file, find_directories_by_name
import csv

def sfm_with_colmap(sfm_pipeline_exe: str, source_dir: str, output_dir: str):
    # 构建COLMAP SfM命令
    sfm_cmd = sfm_pipeline_exe + "  -s " + source_dir + " -o " + output_dir
    print("sfm_cmd: {}".format(sfm_cmd))
    # start = time.time()
    os.system(sfm_cmd)
    # end = time.time()
    # used = end - start
    # return used


def densify_pointcloud(densfy_exe_path: str, project_dir: str, output_dir: str):    
    densify_cmd = "{} -i {} -o {}".format(densfy_exe_path, project_dir, output_dir)
    print("densify_cmd: {}".format(densify_cmd))
    exit_code = os.system(densify_cmd)
    if exit_code != 0:
        logging.error(f"Densification failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:
        logging.info("Densification successful")
        return 0


def get_depth_map(depthAnythingPath: str, project_dir: str, output_dir: str):
    get_depth_cmd = f"python {depthAnythingPath} --img-path {project_dir} --outdir {output_dir}"
    exit_code = os.system(get_depth_cmd)
    if exit_code != 0:
        logging.error(f"Getting depth map failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:
        logging.info("Getting depth map successful")
        return 0


def correct_depth_map(correct_exe_path, project_dir, depth_dir):
    correct_cmd = f"python {correct_exe_path} --base_dir {project_dir} --depths_dir {depth_dir}"
    print(f"correct depth cmd: {correct_cmd}")
    exit_code = os.system(correct_cmd)
    if exit_code != 0:
        logging.error(f"correct depths failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:
        logging.info("correct depths map successful")
        return 0


def run_3dgs(project_dir: str, output_dir: str):
    train_cmd = f"python {train_script_path} -s {project_dir} -m {output_dir}"
    print(f"train cmd: {train_cmd}")
    exit_code = os.system(train_cmd)
    if exit_code != 0:
        logging.error(f"train 3dgs failed with code {exit_code}. Exiting.")
        exit(exit_code)
    else:
        logging.info("train successful")
        return 0

def train_3dgs(args: Namespace):

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    try:
        if(args.websockets):
            network_gui_ws.init(args.ip, args.port)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)

        psnr_list, time_list = training(
            lp.extract(args), 
            op.extract(args), 
            pp.extract(args), 
            args.test_iterations, 
            args.save_iterations,
            args.debug_from, 
            args.websockets,
            args
        )

        return psnr_list, time_list
    except Exception as e:
        logging.exception("train_3dgs failed, continuing pipeline: %s", e)
        # Return safe default lists (same shape as expected) so caller can continue
        return [(0, 0.0)], [(0, 0.0)]


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



class TimeAndMetrics:
    def __init__(
        self,
        sfm_time=0.0,
        densify_time=0.0,
        get_depths_time=0.0,
        correct_depth_time=0.0,
        fast_gs_time=0.0,
        refine_gs_time=0.0,
        sfm_input_num=0,
        sfm_registered_num=0,
        fast_psnr=0.0,
        fast_ssim=0.0,
        fast_lpips=0.0,
        refine_psnr=0.0,
        refine_ssim=0.0,
        refine_lpips=0.0,
    ):
        self.sfm_time = sfm_time
        self.sfm_input_num = sfm_input_num
        self.sfm_registered_num = sfm_registered_num
        self.sfm_registered_ratio = 0.0
        self.densify_time = densify_time
        self.get_depths_time = get_depths_time
        self.correct_depth_time = correct_depth_time
        self.fast_gs_time = fast_gs_time
        self.refine_gs_time = refine_gs_time

        self.total_time = 0.0

        self.fast_psnr = fast_psnr
        self.fast_ssim = fast_ssim
        self.fast_lpips = fast_lpips
        self.refine_psnr = refine_psnr
        self.refine_ssim = refine_ssim
        self.refine_lpips = refine_lpips

    def get_all_time(self):
        self.total_time = (
            self.sfm_time
            + self.densify_time
            + self.get_depths_time
            + self.correct_depth_time
            + self.refine_gs_time
        )
        return self.total_time
    
    def get_sfm_registered_ratio(self):
        if self.sfm_input_num > 0:
            self.sfm_registered_ratio = self.sfm_registered_num / self.sfm_input_num
        else:
            self.sfm_registered_ratio = 0.0
        return self.sfm_registered_ratio



def load_metrics_from_txt(project_dir: str, project_name: str, time_and_metrics_list: dict) -> bool:
    """
    If `project_dir/metrics.txt` exists, parse it into a TimeAndMetrics object,
    insert into `time_and_metrics_list` under `project_name` and return True.
    On any failure return False.
    """
    metrics_path = os.path.join(project_dir, "metrics.txt")
    if not os.path.exists(project_dir):
        return False
    else:
        if not os.path.exists(metrics_path):
            delete_directory(project_dir)
            return False

    try:
        import re
        with open(metrics_path, 'r', encoding='utf-8') as mf:
            content = mf.read()

        parsed_metrics = TimeAndMetrics()

        def _extract_float(pat):
            m = re.search(pat, content)
            return float(m.group(1)) if m else None

        def _extract_int(pat):
            m = re.search(pat, content)
            return int(m.group(1)) if m else None

        v = _extract_float(r"SfM 时间: ([\d\.]+) s")
        if v is not None:
            parsed_metrics.sfm_time = v

        v = _extract_float(r"点云密化时间: ([\d\.]+) s")
        if v is not None:
            parsed_metrics.densify_time = v

        v = _extract_float(r"深度图生成时间: ([\d\.]+) s")
        if v is not None:
            parsed_metrics.get_depths_time = v

        v = _extract_float(r"深度图校正时间: ([\d\.]+) s")
        if v is not None:
            parsed_metrics.correct_depth_time = v

        v = _extract_float(r"快速高斯训练时间: ([\d\.]+) s")
        if v is not None:
            parsed_metrics.fast_gs_time = v

        v = _extract_float(r"精细高斯训练时间: ([\d\.]+) s")
        if v is not None:
            parsed_metrics.refine_gs_time = v

        v = _extract_int(r"输入图片数量: (\d+)")
        if v is not None:
            parsed_metrics.sfm_input_num = v

        v = _extract_int(r"注册图片数量: (\d+)")
        if v is not None:
            parsed_metrics.sfm_registered_num = v

        v = _extract_float(r"SfM 注册率: ([\d\.]+)")
        if v is not None:
            parsed_metrics.sfm_registered_ratio = v

        v = _extract_float(r"快速高斯 PSNR: ([\d\.]+)")
        if v is not None:
            parsed_metrics.fast_psnr = v

        v = _extract_float(r"快速高斯 SSIM: ([\d\.]+)")
        if v is not None:
            parsed_metrics.fast_ssim = v

        v = _extract_float(r"快速高斯 LPIPS: ([\d\.]+)")
        if v is not None:
            parsed_metrics.fast_lpips = v

        v = _extract_float(r"精细高斯 PSNR: ([\d\.]+)")
        if v is not None:
            parsed_metrics.refine_psnr = v

        v = _extract_float(r"精细高斯 SSIM: ([\d\.]+)")
        if v is not None:
            parsed_metrics.refine_ssim = v

        v = _extract_float(r"精细高斯 LPIPS: ([\d\.]+)")
        if v is not None:
            parsed_metrics.refine_lpips = v

        parsed_metrics.total_time = parsed_metrics.get_all_time()

        if "extracted" in project_name:
            project_name = os.path.basename(os.path.dirname(os.path.dirname(project_dir))) + "_" + project_name

        time_and_metrics_list[project_dir] = parsed_metrics
        return True

    except Exception as e:
        logging.error(f"Failed to load metrics from {metrics_path}: {e}")
        try:
            delete_directory(project_dir)
        except Exception:
            pass
        return False


def save_metrics_to_txt(project_dir, project_name, time_and_metrics):
    """
    将 TimeAndMetrics 对象保存为 txt 文件。
    
    Args:
        project_dir (str): 项目目录。
        project_name (str): 项目名称。
        time_and_metrics (TimeAndMetrics): 时间和指标对象。
    """
    metrics_txt_path = os.path.join(project_dir, "metrics.txt")
    try:
        with open(metrics_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Project: {project_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # 写入时间信息
            f.write("时间统计 (单位: 秒)\n")
            f.write("-" * 60 + "\n")
            f.write(f"SfM 时间: {time_and_metrics.sfm_time:.2f} s\n")
            f.write(f"点云密化时间: {time_and_metrics.densify_time:.2f} s\n")
            f.write(f"深度图生成时间: {time_and_metrics.get_depths_time:.2f} s\n")
            f.write(f"深度图校正时间: {time_and_metrics.correct_depth_time:.2f} s\n")
            f.write(f"快速高斯训练时间: {time_and_metrics.fast_gs_time:.2f} s\n")
            f.write(f"精细高斯训练时间: {time_and_metrics.refine_gs_time:.2f} s\n")
            f.write(f"总耗时: {time_and_metrics.total_time:.2f} s\n\n")

            # 写入SfM统计
            f.write("SfM 统计\n")
            f.write("-" * 60 + "\n")
            f.write(f"输入图片数量: {time_and_metrics.sfm_input_num}\n")
            f.write(f"注册图片数量: {time_and_metrics.sfm_registered_num}\n\n")
            f.write(f"SfM 注册率: {time_and_metrics.get_sfm_registered_ratio():.2f}\n\n")
            
            # 写入质量指标
            f.write("质量指标\n")
            f.write("-" * 60 + "\n")
            f.write(f"快速高斯 PSNR: {time_and_metrics.fast_psnr:.2f}\n")
            f.write(f"快速高斯 SSIM: {time_and_metrics.fast_ssim:.2f}\n")
            f.write(f"快速高斯 LPIPS: {time_and_metrics.fast_lpips:.2f}\n")
            f.write(f"精细高斯 PSNR: {time_and_metrics.refine_psnr:.2f}\n")
            f.write(f"精细高斯 SSIM: {time_and_metrics.refine_ssim:.2f}\n")
            f.write(f"精细高斯 LPIPS: {time_and_metrics.refine_lpips:.2f}\n")

        logging.info(f"成功保存metrics至: {metrics_txt_path}")
    except Exception as e:
        logging.error(f"保存metrics失败: {e}")


def save_metrics_to_csv_sorted(metrics_dict, output_path, fieldnames=None):
    """
    将 TimeAndMetrics 字典按键排序后保存为 CSV 文件。
    
    Args:
        metrics_dict (dict): 键为 str，值为 TimeAndMetrics 对象。
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
    if fieldnames is None:
        fieldnames = ['id'] + list(vars(first_obj).keys())
    else:
        # 确保 'id' 在第一列
        if 'id' not in fieldnames:
            fieldnames = ['id'] + fieldnames

    try:
        with open(output_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 3. 按 key (即 id) 的字母顺序排序
            for key in sorted(metrics_dict.keys()):
                obj = metrics_dict[key]
                
                # 更新总时间
                obj.total_time = (
                    obj.sfm_time + obj.densify_time + obj.get_depths_time + 
                    obj.correct_depth_time + obj.refine_gs_time
                )
                
                # 转换为字典并写入
                row = vars(obj).copy()
                row['id'] = key
                
                # 对浮点数进行格式化，只保留2位小数
                for key_name in row:
                    if isinstance(row[key_name], float):
                        row[key_name] = f"{row[key_name]:.2f}"
                
                writer.writerow(row)
                
        print(f"成功按 ID 排序并保存至: {output_path}")

    except Exception as e:
        print(f"保存失败: {e}")

if __name__ == '__main__':

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * (i + 1) for i in range(20)])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 7000, 10000, 15000, 20000, 25000, 30_000, 35000, 40000])
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 20000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000, 20000])
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--cams", type=int, default=10)
    # parser.add_argument("--websockets", action='store_true', default=False)

    parser.add_argument("--pipeline", nargs="+", type=str, default=["sfm"])
    # parser.add_argument("--skyball", action='store_true', default=False, help='添加天空球到点云')
    # parser.add_argument("--skyball_points", type=int, default=10000, help='天空球点数')

    parser.add_argument("--dataset_root", "-ds", type=str, default="dataset", help='数据集根目录')
    parser.add_argument("--output_name", "-o", type=str, default="output-sfm-aliked-threshold-acc")
    parser.add_argument("--metrics_file_name", "-mf", type=str, default="metric_sfm-aliked-threshold-acc.csv", help='保存所有数据集时间和指标的csv文件名')

    args = parser.parse_args()
    # args.save_iterations.append(args.iterations)

    dataset_path = find_directories_by_name(args.dataset_root, "input")
    if not dataset_path:
        print("没有找到任何数据集，程序退出。")
        sys.exit(0)

    dataset_path = sorted(dataset_path)
    
 
    current_path = resource_path()
    print(f"current path: {current_path}")

    # =========================处理每个数据集===================================================================================
    time_and_metrics_list = {}

    for dataset_dir in dataset_path:
        data_path = os.path.dirname(dataset_dir)        
        project_name = os.path.basename(data_path)
        project_dir = os.path.join(os.path.dirname(dataset_dir), args.output_name)
        print(f"Processing dataset at: {os.path.dirname(dataset_dir)}")

        metrics_path = os.path.join(project_dir, "metrics.txt")
        # if load_metrics_from_txt(project_dir, project_name, time_and_metrics_list):
        #     print(f"Loaded existing metrics for {project_name} from {metrics_path}; skipping reprocessing.")
        #     continue

        os.makedirs(project_dir, exist_ok=True)        

        args.source_path = project_dir
        args.model_path = os.path.join(project_dir, "3dgs_model")
        os.makedirs(args.model_path, exist_ok=True)

        time_and_metrics = TimeAndMetrics()

        # ==========================Colmap-SFM===============================================================================
        sfm_start = time.time()
        if "sfm" in args.pipeline:
            # sfm_command = os.path.join(current_path, "run-sfm-20260311/run-sfm.exe") 
            # sfm_command = "python sfm-colmap-3.14.0.py --alg acc "
            # sfm_command = "python sfm-v7.py --alg acc --max_feature_num 8192  -sms acc "
            # sfm_command = "python sfm-v7.py --alg acc --max_feature_num 8192  -sms sequential "
            # sfm_command = "python sfm-v6.py --alg acc --max_feature_num 2048 -splg --clean "
            sfm_command = "python sfm-colmap-v4.py --clean"
            # sfm_command = sfm_pipeline_exe + " --alg acc -sms " + args.sift_match_strategy + " -so " + str(args.sequential_overlap) + " "
            sfm_output_dir = project_dir
            sfm_with_colmap(sfm_command, data_path, sfm_output_dir)   

        sfm_used = time.time() - sfm_start
        print("Sfm used: {:.2f}s".format(sfm_used))

        input_img_num = count_image_files(dataset_dir)
        print(f"输入图片数量: {input_img_num}")
        registered_img_num = count_image_files(os.path.join(project_dir, "images"))
        print(f"注册图片数量: {registered_img_num}")

        time_and_metrics.sfm_time = sfm_used
        time_and_metrics.sfm_input_num = input_img_num
        time_and_metrics.sfm_registered_num = registered_img_num

        # ===========================Gaussian Splatting=========================================================================
        train_start = time.time()
        psnr_list = None
        time_list = None
        if "3dgs" in args.pipeline:
            train_script_path = os.path.join(current_path, "train.py")
            output_3dgs_path = args.model_path
            psnr_list, time_list = train_3dgs(args)

        train_used = time.time() - train_start
        print(f"training 3dgs used time: {train_used:.2f}s")
        if time_list is not None and psnr_list is not None:
            time_and_metrics.fast_gs_time = time_list[0][1]
            time_and_metrics.refine_gs_time = time_list[-1][1]
            time_and_metrics.fast_psnr = psnr_list[0][1]
            time_and_metrics.refine_psnr = psnr_list[-1][1]

        time_and_metrics.total_time = time_and_metrics.get_all_time()

        # ===================================================================================================================
        print(f"Total pipeline time for {project_name}: {time_and_metrics.total_time:.2f} s")
        print(f"SfM time: {time_and_metrics.sfm_time:.2f} s")
        print(f"Densify time: {time_and_metrics.densify_time:.2f} s")
        print(f"Get Depths time: {time_and_metrics.get_depths_time:.2f} s")
        print(f"Correct Depth time: {time_and_metrics.correct_depth_time:.2f} s")
        print(f"Fast GS time: {time_and_metrics.fast_gs_time:.2f} s")
        print(f"Refine GS time: {time_and_metrics.refine_gs_time:.2f} s")
        print(f"Fast PSNR: {time_and_metrics.fast_psnr:.2f}")
        print(f"Refine PSNR: {time_and_metrics.refine_psnr:.2f}")

        time_and_metrics_list[project_dir] = time_and_metrics

        # ==========================保存metrics到txt文件=========================================================================
        save_metrics_to_txt(project_dir, project_name, time_and_metrics)

        # ==========================清理无用文件===============================================================================
        delete_directory(os.path.join(project_dir, "dense"))
        delete_directory(os.path.join(project_dir, "depths"))
        delete_directory(os.path.join(project_dir, "distorted"))
        delete_directory(os.path.join(project_dir, "stereo"))
        delete_file(os.path.join(project_dir, "run-colmap-geometric.sh"))
        delete_file(os.path.join(project_dir, "run-colmap-photometric.sh"))
    # =========================总结所有数据集的时间和指标===========================================================================
    print("\n=== Summary of All Datasets ===")

    # 计算均值
    if time_and_metrics_list:
        avg_metrics = TimeAndMetrics()
        
        # 计算所有指标的均值
        num_datasets = len(time_and_metrics_list)
        
        avg_metrics.sfm_time = sum(m.sfm_time for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.sfm_input_num = sum(m.sfm_input_num for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.sfm_registered_num = sum(m.sfm_registered_num for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.sfm_registered_ratio = sum(m.get_sfm_registered_ratio() for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.densify_time = sum(m.densify_time for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.get_depths_time = sum(m.get_depths_time for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.correct_depth_time = sum(m.correct_depth_time for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.fast_gs_time = sum(m.fast_gs_time for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.refine_gs_time = sum(m.refine_gs_time for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.fast_psnr = sum(m.fast_psnr for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.fast_ssim = sum(m.fast_ssim for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.fast_lpips = sum(m.fast_lpips for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.refine_psnr = sum(m.refine_psnr for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.refine_ssim = sum(m.refine_ssim for m in time_and_metrics_list.values()) / num_datasets
        avg_metrics.refine_lpips = sum(m.refine_lpips for m in time_and_metrics_list.values()) / num_datasets
        
        # 计算总耗时均值
        avg_metrics.total_time = avg_metrics.get_all_time()
        
        # 打印均值信息
        print("\n平均值统计:")
        print("-" * 60)
        print(f"平均 SfM 时间: {avg_metrics.sfm_time:.2f} s")
        print(f"平均点云密化时间: {avg_metrics.densify_time:.2f} s")
        print(f"平均深度图生成时间: {avg_metrics.get_depths_time:.2f} s")
        print(f"平均深度图校正时间: {avg_metrics.correct_depth_time:.2f} s")
        print(f"平均快速高斯训练时间: {avg_metrics.fast_gs_time:.2f} s")
        print(f"平均精细高斯训练时间: {avg_metrics.refine_gs_time:.2f} s")
        print(f"平均总耗时: {avg_metrics.total_time:.2f} s")
        print(f"平均输入图片数量: {avg_metrics.sfm_input_num:.2f}")
        print(f"平均注册图片数量: {avg_metrics.sfm_registered_num:.2f}")
        print(f"平均 SfM 注册率: {avg_metrics.get_sfm_registered_ratio():.2f}")
        print(f"平均快速高斯 PSNR: {avg_metrics.fast_psnr:.2f}")
        print(f"平均快速高斯 SSIM: {avg_metrics.fast_ssim:.2f}")
        print(f"平均快速高斯 LPIPS: {avg_metrics.fast_lpips:.2f}")
        print(f"平均精细高斯 PSNR: {avg_metrics.refine_psnr:.2f}")
        print(f"平均精细高斯 SSIM: {avg_metrics.refine_ssim:.2f}")
        print(f"平均精细高斯 LPIPS: {avg_metrics.refine_lpips:.2f}")
        
        # 将平均值添加到列表中
        time_and_metrics_list['Average'] = avg_metrics
    
    
    custom_fieldnames = [
        'sfm_time', 'densify_time', 'get_depths_time', 'correct_depth_time',
        'fast_gs_time', 'refine_gs_time', 'total_time', 'sfm_input_num', 'sfm_registered_num', "sfm_registered_ratio",
        'fast_psnr', 'fast_ssim', 'fast_lpips',
        'refine_psnr', 'refine_ssim', 'refine_lpips'
    ]
    
    save_metrics_to_csv_sorted(time_and_metrics_list, os.path.join(args.dataset_root, args.metrics_file_name), fieldnames=custom_fieldnames)