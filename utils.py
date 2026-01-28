
import sys
import os
import subprocess
import logging
from typing import Optional
import shutil
import logging

def check_operating_system():
    """
    判断当前操作系统类型（Windows/Linux）
    返回值：字符串 'Windows' 或 'Linux' 或 'Unknown'
    """
    # 方法1：使用 sys.platform（推荐）
    platform = sys.platform
    print(f"sys.platform 返回值: {platform}")
    
    if platform.startswith('win'):
        os_type = 'Windows'
    elif platform in ('linux', 'linux2'):  # linux2 是 Python2 的返回值，linux 是 Python3 的返回值
        os_type = 'Linux'
    else:
        os_type = 'Unknown'

    return os_type


# def resource_path() -> str:
#     """Get absolute path for packaged or development scripts."""
#     try:
#         base_path = sys._MEIPASS
#     except AttributeError:
#         base_path = os.path.dirname(os.path.abspath(__file__))
#     return base_path

def run_subprocess(cmd: list, logger: Optional[logging.Logger] = None) -> int:
    """
    Run a subprocess command, print stdout/stderr in real-time and save to log file.
    Windows compatible. Uses the provided logger (per-folder) if given.
    
    Returns:
        0 if the command succeeded, -1 if it failed or crashed.
    """
    if logger is None:
        logger = logging.getLogger("sfm")

    # logger.info(f"Running command: {' '.join(cmd)}")

    # Check if the logger has a StreamHandler attached that writes to stdout (robust check)
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, getattr(sys, "__stdout__", None))
        for h in logger.handlers
    )

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            shell=False  # Windows, using list command
        )
    except Exception as e:
        logger.error(f"Error: Failed to start process: {e}")
        return -1

    # Real-time printing and logging
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # If logger writes to stdout, rely on it (formatted). Otherwise, print to terminal (flush) to guarantee visibility.
                if has_stdout_handler:
                    logger.info(line)
                else:
                    print(line, flush=True)
                    logger.info(line)
    except Exception as e:
        logger.error(f"Error reading process output: {e}")
        try:
            process.kill()
        except:
            logger.error("Error: Failed to kill process after output read error.")
            pass
        return -1

    process.wait()

    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}.")
        return -1
    else:
        logger.info("Command completed successfully.")
        return 0



# def run_subprocess(cmd: list, log_path: str):
#     """
#     Run a subprocess command, print stdout/stderr in real-time and save to log file.
#     Windows compatible.
#     """
#     # logger.info(f"Running command: {' '.join(cmd)}")
    
#     with open(log_path, "w", encoding="utf-8") as log_file:
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             bufsize=1,  # line-buffered
#             shell=False  # Windows, using list command
#         )

#         # Real-time printing and logging
#         for line in process.stdout:
#             line = line.rstrip()
#             if line:
#                 print(line)
#                 log_file.write(line + "\n")
#         process.wait()

#     if process.returncode != 0:
#         log_file.write(f"Command failed with code {process.returncode}. See {log_path} for details.")
#         raise subprocess.CalledProcessError(process.returncode, cmd)


def configure_logger(output_path: str, level: int, logger_name: Optional[str] = None) -> logging.Logger:
    """Create or reconfigure a named logger for each run to avoid duplicate handlers.

    If logger_name is provided, a distinct logger will be used for that data folder, enabling
    separate log files and isolation between iterations.
    """
    if logger_name is None:
        logger_name = "sfm"

    # Normalize level: if user passed 0 or invalid small number, default to INFO
    if level <= 0:
        level = logging.INFO
    else:
        level = int(level)

    logger = logging.getLogger(logger_name)
    # Remove existing handlers to avoid duplicate logs when reconfiguring
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(level)
    logger.propagate = False

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler -> explicitly send to stdout so it appears in terminal
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

    # File handler (overwrite per run)
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "run-sfm.log")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)

    # Emit a small verification message (both print and logger) so terminal visibility can be tested
    print(f"[sfm] Logger '{logger_name}' configured for '{output_path}' (level={level})", flush=True)
    logger.info(f"Logger '{logger_name}' configured (level={level}) for '{output_path}'")

    return logger


def get_subfolders(parent_dir: str) -> list:
    """Return a list of subfolder paths in the given parent directory."""
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Path not found or not a directory: {parent_dir}")
    subfolders = []
    for name in os.listdir(parent_dir):
        sub_path = os.path.join(parent_dir, name)
        if os.path.isdir(sub_path):
            subfolders.append(sub_path)
    return subfolders


def get_largest_subfolder(parent_dir: str) -> Optional[str]:
    """Return the subfolder with the largest total file size."""
    if not os.path.isdir(parent_dir):
        raise ValueError(f"Path not found or not a directory: {parent_dir}")
    max_size = -1
    largest_subfolder = None
    for name in os.listdir(parent_dir):
        sub_path = os.path.join(parent_dir, name)
        if not os.path.isdir(sub_path):
            continue
        total_size = 0
        for root, _, files in os.walk(sub_path):
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except OSError:
                    pass
        if total_size > max_size:
            max_size = total_size
            largest_subfolder = sub_path
    return largest_subfolder


def count_images_in_dir(folder_path: str) -> int:
    """
    统计指定文件夹路径下的图像文件数量（仅当前文件夹，不包含子文件夹）
    """
    # 定义常见的图像文件扩展名（转小写，方便统一判断）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

    image_count = 0

    if not os.path.exists(folder_path):
        print(f"错误：文件夹路径 '{folder_path}' 不存在！")
        return 0
    if not os.path.isdir(folder_path):
        print(f"错误：'{folder_path}' 不是一个有效的文件夹路径！")
        return 0

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_extensions:
                image_count += 1

    return image_count



def count_images_in_dir_recursive(root_dir, image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

    from pathlib import Path
    root = Path(root_dir)
    # 转为小写便于比较
    image_extensions = {ext.lower() for ext in image_extensions}
    
    count = 0
    for file in root.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            count += 1
    return count


def clear_folder(folder_path):
    """
    清空指定文件夹内的所有内容（包括文件和子文件夹）
    
    Args:
        folder_path (str): 目标文件夹的路径
        
    Raises:
        FileNotFoundError: 如果指定的文件夹路径不存在
        PermissionError: 如果没有操作该文件夹的权限
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    # 检查是否是文件夹（而非文件）
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} 不是一个有效的文件夹")
    
    # 遍历文件夹内的所有内容
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        
        try:
            # 如果是文件或符号链接，直接删除
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"已删除文件: {item_path}")
            # 如果是文件夹，递归删除
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"已删除文件夹及其内容: {item_path}")
        except Exception as e:
            print(f"删除 {item_path} 时出错: {e}")