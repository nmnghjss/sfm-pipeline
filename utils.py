
import ctypes
import sys
import os
import subprocess
import logging
from typing import Optional
import shutil
from typing import List

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
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif','.webp'}

    from pathlib import Path
    root = Path(root_dir)
    # 转为小写便于比较
    image_extensions = {ext.lower() for ext in image_extensions}
    
    count = 0
    images_list = []
    for file in root.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            count += 1
            images_list.append(file)
    return count, images_list


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


def delete_directory(path):
    """递归删除目录及其所有内容"""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        else:
            return False
    except Exception as e:
        return False


def delete_file(path):
    """删除单个文件"""
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
        else:
            return False
    except Exception as e:
        return False

def copy_file(src, dst):
    try:
        shutil.copy2(src, dst)  # copy2会保留元数据（修改时间等）
        return True
    except Exception as e:
        return False
    

def get_first_level_subdirs(parent_dir: str) -> List[str]:
    """
    获取指定目录下的第一级子目录（不递归）

    返回的是完整路径列表
    """
    subdirs = []

    with os.scandir(parent_dir) as it:
        for entry in it:
            if entry.is_dir():
                subdirs.append(entry.path)

    return subdirs

def count_image_files(folder_path: str) -> int:
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


def find_directories_by_name(root_folder, target_name):
    """
    递归搜索指定目录下所有名为 target_name 的子目录。
    
    Args:
        root_folder (str): 起始搜索的根目录路径
        target_name (str): 要寻找的目录名称（例如 "images" 或 "output"）
        
    Returns:
        list: 包含所有匹配目录绝对路径的列表
    """
    matched_directories = []

    # os.walk 会生成三元组 (当前路径, 当前路径下的子目录列表, 当前路径下的文件列表)
    for root, dirs, files in os.walk(root_folder):
        # 我们在 dirs 列表中寻找匹配的名称
        if target_name in dirs:
            # 拼接出完整的绝对路径
            full_path = os.path.abspath(os.path.join(root, target_name))
            matched_directories.append(full_path)
            
            # 优化提示：
            # 如果你确定 target_name 目录下面不会再有同名的子目录，
            # 可以通过修改 dirs 来阻止 os.walk 继续深入这个分支，提高效率：
            # dirs.remove(target_name) 

    return matched_directories


def move_folder(source_dir: str, target_parent_dir: str) -> bool:
    """
    将源文件夹移动到指定的目标父文件夹中
    
    Args:
        source_dir: 源文件夹的路径（绝对/相对路径均可）
        target_parent_dir: 目标父文件夹的路径（源文件夹会被移动到这个文件夹下）
    
    Returns:
        bool: 移动成功返回True，失败返回False
    
    Raises:
        无（内部捕获所有常见异常并打印提示，返回False）
    """
    # 1. 规范化路径（处理末尾斜杠、相对路径等问题）
    source_dir = os.path.abspath(source_dir)
    target_parent_dir = os.path.abspath(target_parent_dir)
    
    # 2. 获取源文件夹的名称（用于拼接最终目标路径）
    source_folder_name = os.path.basename(source_dir)
    # 最终目标路径：目标父文件夹 + 源文件夹名
    final_target_dir = os.path.join(target_parent_dir, source_folder_name)
    
    try:
        # 3. 校验源目录是否存在且是文件夹
        if not os.path.exists(source_dir):
            print(f"错误：源文件夹 '{source_dir}' 不存在！")
            return False
        if not os.path.isdir(source_dir):
            print(f"错误：'{source_dir}' 不是一个文件夹！")
            return False
        
        # 4. 校验目标父目录是否存在，不存在则创建
        if not os.path.exists(target_parent_dir):
            print(f"目标父文件夹 '{target_parent_dir}' 不存在，正在创建...")
            os.makedirs(target_parent_dir, exist_ok=True)  # exist_ok=True 避免重复创建报错
        
        # 5. 检查目标路径是否已存在（避免覆盖）
        if os.path.exists(final_target_dir):
            print(f"错误：目标路径 '{final_target_dir}' 已存在，无法移动（避免覆盖）！")
            return False
        
        # 6. 执行文件夹移动（核心操作）
        shutil.move(source_dir, target_parent_dir)
        print(f"成功！文件夹已从 '{source_dir}' 移动到 '{final_target_dir}'")
        return True
    
    except PermissionError:
        print(f"错误：权限不足，无法移动文件夹 '{source_dir}'！")
        return False
    except Exception as e:
        print(f"移动失败：未知错误 - {str(e)}")
        return False
    

# Helper function to convert Chinese paths to short paths on Windows
def get_short_path_name(long_path):
    """
    Convert a long path to its short (8.3) path format on Windows.
    This helps handle paths with non-ASCII characters (like Chinese).
    
    Args:
        long_path: The long path to convert
        
    Returns:
        Short path if on Windows, original path otherwise
    """
    if sys.platform != 'win32':
        return long_path
    
    try:
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.kernel32.GetShortPathNameW(long_path, buffer, len(buffer)):
            short_path = buffer.value
            logger.debug(f"Converted path: {long_path} -> {short_path}")
            return short_path
    except Exception as e:
        logger.warning(f"Failed to convert path to short format: {e}")
    
    return long_path  # Fallback to the original path if conversion fails
