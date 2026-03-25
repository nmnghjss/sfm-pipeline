import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
from torchvision.io import read_image as torchvision_read_image
from torchvision.transforms import functional as F
import logging
import psutil
import gc
import PIL

class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """Move batch (dict) to device"""

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    # Use np.fromfile to handle non-ASCII paths (including Chinese characters) on Windows
    # cv2.imread doesn't handle UTF-8 paths well on Windows
    image_data = np.fromfile(str(path), dtype=np.uint8)
    if grayscale:
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]  # BGR to RGB
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    
    # t3 = time.time()
    # return torch.tensor(image / 255.0, dtype=torch.float)

    # 使用更高效的方法：先转为float32，然后直接共享内存（零拷贝）
    image_float32 = image.astype(np.float32, copy=False)  # copy=False避免不必要的复制
    # result = torch.from_numpy(image_float32).div_(255.0)  # div_是就地操作，更高效
    result = torch.from_numpy(image_float32)
    # print("convert time: ", time.time() - t3)
    return result


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale

# import time
def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    # t1 = time.time()
    image = read_image(path)
    # print("load time: ", time.time() - t1)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)



def safe_load_image(image_path: str, logger: logging.Logger = None) -> torch.Tensor:
    try:
        # 1. 安全读取：支持中文路径，且使用 context manager 确保文件句柄即时关闭
        with open(image_path, 'rb') as f:
            # 使用 np.frombuffer 替代 fromfile，它直接操作二进制流，更轻量
            logger.info(f"start to load image buff: {image_path}")
            image_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # 2. 解码
        logger.info(f"start to decode image: {image_path}")
        img_cpu = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img_cpu is None:
            return None
 
        # CPU 处理流
        logger.info(f"start to convert color {image_path}")
        img_rgb = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2RGB)
        logger.info(f"start to convert uint8 to float: {image_path}")
        img_rgb = img_rgb.astype(np.float32)
        logger.info(f"start to convert numpy array to torch tensor {image_path}")
        tensor = torch.from_numpy(img_rgb)

        # 维度变换 (C, H, W)
        tensor = tensor.permute(2, 0, 1).contiguous()
        
        return tensor

    except Exception as e:
        logger.info(f"读取失败: {image_path}, 错误: {e}")
        return None
    finally:
        # 强制少量垃圾回收（可选，仅在频繁被杀时开启）
        gc.collect()



def load_image_opencv_tensor(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    *,
    grayscale: bool = False,
    channels_first: bool = True,
    dtype: torch.dtype = torch.uint8,
    pin_memory: bool = True,
    non_blocking: bool = True,
    gpu_decode_backend: str = "dali",
    allow_gpu_fallback_to_cpu_decode: bool = False,
) -> torch.Tensor:
    """
    Read an image and return a torch Tensor on CPU or GPU.

    CPU:
    - Uses OpenCV CPU decode (imdecode) + torch.from_numpy (zero-copy when possible).

    GPU:
    - IMPORTANT: OpenCV-Python `cv2.cuda` does NOT provide GPU image *decoding* for still images
      (no cuda::imread/imdecode). It can only upload/download and run CUDA ops.
    - Therefore, to *actually decode on GPU*, this function uses NVIDIA DALI (nvJPEG) when
      `gpu_decode_backend="dali"`. If DALI is unavailable, we either raise (default) or fall
      back to CPU decode + async H2D copy if `allow_gpu_fallback_to_cpu_decode=True`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No image at path {p}.")

    target = torch.device(device) if not isinstance(device, torch.device) else device
    if target.type == "cpu":
        # Robust Windows path handling (incl. non-ASCII).
        image_data = np.fromfile(str(p), dtype=np.uint8)
        if grayscale:
            img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image at {p}.")

        if not grayscale:
            # cv2.cvtColor produces a contiguous array (avoids negative-stride slicing issues).
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if channels_first:
            if img.ndim == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            elif img.ndim == 2:
                img = np.ascontiguousarray(img[None, ...])
            else:
                raise ValueError(f"Not an image: {img.shape}")
        else:
            img = np.ascontiguousarray(img)

        cpu_tensor = torch.from_numpy(img)
        if dtype != cpu_tensor.dtype:
            cpu_tensor = cpu_tensor.to(dtype=dtype)
        return cpu_tensor

    # GPU: prefer true GPU decode via DALI (nvJPEG).
    if gpu_decode_backend.lower() == "dali":
        try:
            from nvidia.dali import fn, pipeline_def  # type: ignore
            import nvidia.dali.plugin.pytorch as dali_torch  # type: ignore
        except Exception as e:
            if not allow_gpu_fallback_to_cpu_decode:
                raise RuntimeError(
                    "GPU decode requested, but NVIDIA DALI is not available. "
                    "Install DALI (with nvJPEG) or set allow_gpu_fallback_to_cpu_decode=True."
                ) from e
            # fallback to CPU decode below
        else:
            # DALI returns HWC uint8 on GPU by default; we'll permute if needed.
            @pipeline_def(batch_size=1, num_threads=1, device_id=target.index or 0)
            def _pipe(file_path: str):
                jpegs, _ = fn.readers.file(files=[file_path], name="reader")
                # "mixed" uses CPU for some stages and GPU for decode/resize; "gpu" is GPU only.
                out = fn.decoders.image(jpegs, device="mixed", output_type=fn.types.RGB)
                return out

            pipe = _pipe(str(p))
            pipe.build()
            (out,) = pipe.run()
            img_hwc = dali_torch.feed_ndarray(out, torch.empty(out.shape(), device="cuda"))
            # feed_ndarray returns a CUDA tensor with DALI memory copied into it (fast, but not zero-copy).
            # Shape: HWC, uint8, RGB
            if channels_first:
                img = img_hwc.permute(2, 0, 1).contiguous()
            else:
                img = img_hwc.contiguous()
            if grayscale:
                # Convert to grayscale on GPU if requested (simple luma approximation).
                if channels_first:
                    r, g, b = img[0:1], img[1:2], img[2:3]
                    img = (0.299 * r + 0.587 * g + 0.114 * b).to(dtype=torch.uint8)
                else:
                    r, g, b = img[..., 0:1], img[..., 1:2], img[..., 2:3]
                    img = (0.299 * r + 0.587 * g + 0.114 * b).to(dtype=torch.uint8)
            if dtype != img.dtype:
                img = img.to(dtype=dtype)
            return img

    # GPU fallback: CPU decode + pinned async H2D copy (fastest available without DALI).
    image_data = np.fromfile(str(p), dtype=np.uint8)
    if grayscale:
        img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image at {p}.")
    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channels_first:
        if img.ndim == 3:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
        elif img.ndim == 2:
            img = np.ascontiguousarray(img[None, ...])
        else:
            raise ValueError(f"Not an image: {img.shape}")
    else:
        img = np.ascontiguousarray(img)

    cpu_tensor = torch.from_numpy(img)
    if dtype != cpu_tensor.dtype:
        cpu_tensor = cpu_tensor.to(dtype=dtype)
    if pin_memory and cpu_tensor.device.type == "cpu":
        try:
            cpu_tensor = cpu_tensor.pin_memory()
        except RuntimeError:
            pass
    return cpu_tensor.to(device=target, non_blocking=non_blocking)



def get_process_memory_mb():
    """获取当前进程的内存占用（单位 MB）"""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)   # 物理内存
    vms_mb = mem_info.vms / (1024 * 1024)   # 虚拟内存
    return rss_mb, vms_mb

def tensor_memory_mb(tensor):
    """计算张量占用的内存（单位 MB）"""
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / (1024 * 1024)

def load_image_use_torchvision(
    path: Path, resize: int = None, logger: logging.Logger = None, debug_mode = False, **kwargs
) -> torch.Tensor:
    if logger is None:
        logger = logging.getLogger(__name__)

    # 打印初始内存状态
    if debug_mode:
        rss_before, vms_before = get_process_memory_mb()
        logger.info(f"Initial memory: RSS={rss_before:.2f} MB, VMS={vms_before:.2f} MB")
        # 直接读取为张量 (CHW格式，uint8类型)
        logger.info(f"to load image: {path}")

    image = torchvision_read_image(str(path))

    # 读取后内存
    if debug_mode:
        rss_after_load, vms_after_load = get_process_memory_mb()
        tensor_mb = tensor_memory_mb(image)
        logger.info(f"{path} loaded. Tensor size: {list(image.shape)}, memory: {tensor_mb:.2f} MB")
        logger.info(f"Memory after load: RSS={rss_after_load:.2f} MB, VMS={vms_after_load:.2f} MB")

    if resize is not None:
        logger.info(f"Resizing image to {resize}")
        image_resized = F.resize(image, resize, **kwargs)
        if debug_mode:
            rss_after_resize, vms_after_resize = get_process_memory_mb()
            resized_tensor_mb = tensor_memory_mb(image_resized)
            logger.info(f"Resized tensor memory: {resized_tensor_mb:.2f} MB")
            logger.info(f"Memory after resize: RSS={rss_after_resize:.2f} MB, VMS={vms_after_resize:.2f} MB")
        return image_resized
    else:
        return image

def load_image_use_PIL(path: Path, resize: int = None, logger: logging.Logger = None, **kwargs):

    if logger is None:
        logger = logging.getLogger(__name__)

    # 打印初始内存状态
    rss_before, vms_before = get_process_memory_mb()
    logger.info(f"Initial memory: RSS={rss_before:.2f} MB, VMS={vms_before:.2f} MB")

    # 直接读取为张量 (CHW格式，uint8类型)
    logger.info(f"to load image: {path}")

    pil_image = PIL.Image.open(str(path))
    image_tensor = torch.from_numpy(np.array(pil_image))
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()

    # 读取后内存
    rss_after_load, vms_after_load = get_process_memory_mb()
    tensor_mb = tensor_memory_mb(image_tensor)
    logger.info(f"{path} loaded. Tensor size: {list(image_tensor.shape)}, memory: {tensor_mb:.2f} MB")
    logger.info(f"Memory after load: RSS={rss_after_load:.2f} MB, VMS={vms_after_load:.2f} MB")

    if resize is not None:
        image_resized = F.resize(image_tensor, resize, **kwargs)
        return image_resized
    else:
        return image_tensor

class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats


def match_pair(
    extractor,
    matcher,
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: str = "cpu",
    **preprocess,
):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    feats0 = extractor.extract(image0, **preprocess)
    feats1 = extractor.extract(image1, **preprocess)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    return feats0, feats1, matches01
