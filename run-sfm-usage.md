# run-sfm.exe 使用说明

## 1. 程序简介

`run-sfm.exe` 是一个基于 COLMAP 的 Structure-from-Motion（SfM）命令行程序，用于完成以下流程：

1. 读取输入图像
2. 初始化 COLMAP 数据库
3. 特征提取
4. 特征匹配
5. 视图图校准
6. 增量、全局或分层式建图
7. 可选的位姿先验约束、匹配过滤、图像去畸变和结果整理
8. 输出 COLMAP 稀疏模型和运行日志

程序采用 `onedir` 方式打包。发布时必须保留整个 `run-sfm` 目录结构，不能只复制 `run-sfm.exe`。

## 2. 发布目录结构

推荐目录结构如下：

```text
run-sfm/
├── run-sfm.exe
└── _internal/
    ├── checkpoints/
    ├── Release-colmap-4.2.0-dev-wl-260819/
    │   ├── colmap.exe
    │   ├── onnxruntime.dll
    │   ├── onnxruntime_providers_cuda.dll
    │   ├── cudart64_12.dll
    │   ├── cudnn64_9.dll
    │   └── 其他 COLMAP 运行库
    └── 其他 Python 运行库
```

请不要删除以下目录：

```text
_internal\Release-colmap-4.2.0-dev-wl-260819
_internal\checkpoints
```

其中：

- `Release-colmap-4.2.0-dev-wl-260819`：COLMAP 可执行文件及其 CUDA、ONNX Runtime 等运行库。
- `checkpoints`：特征提取和匹配使用的 ONNX 模型、词袋树文件等资源。

## 3. 输入数据目录

使用 `--source_path` 指定数据根目录。最基本的目录结构如下：

```text
scene/
├── input/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── image_0003.jpg
└── calibration.json       # 可选
```

也可以按子目录组织图像，例如：

```text
scene/
├── input/
│   ├── camera_1/
│   │   ├── image_0001.jpg
│   │   └── image_0002.jpg
│   └── camera_2/
│   |   ├── image_0001.jpg
│   |   └── image_0002.jpg
│   ├── camera_3/
│   │   ├── image_0001.jpg
│   │   └── image_0002.jpg
└── calibration.json       # 可选
```

支持的图像扩展名主要包括：

```text
.jpg .jpeg .png .bmp .tiff .tif .webp
```

## 4. 基本用法

在 PowerShell 中执行：

```powershell
.\run-sfm.exe `
  --source_path "E:\qiyu-test\test0819" `
  --output_path "output-debug"
```

输出目录会被解析为：

```text
E:\qiyu-test\test0819\output-debug
```

当 `--output_path` 使用绝对路径时，将直接使用该路径：

```powershell
.\run-sfm.exe `
  --source_path "E:\qiyu-test\test0819" `
  --output_path "E:\sfm-results\scene01"
```

## 5. 常用运行示例

### 5.1 SIFT + 词袋树匹配 + 全局式建图

```powershell
.\run-sfm.exe `
  --source_path "E:\data\scene01" `
  --output_path "output-sift" `
  --feature_type SIFT `
  --match_strategy vocab_tree `
  --mapper global
```

### 5.2 不使用 GPU

```powershell
.\run-sfm.exe `
  --source_path "E:\data\scene01" `
  --output_path "output-cpu" `
  --no_gpu
```

`--no_gpu` 会将 COLMAP 的特征提取、匹配和建图 GPU 参数设置为关闭。若使用 CUDA 版本 COLMAP，默认情况下程序会尝试使用 GPU。

### 5.3 使用先验位姿生成匹配对

```powershell
.\run-sfm.exe `
  --source_path "E:\qiyu-test\test0819" `
  --output_path "output-pose-prior" `
  --pose_prior "prior-sparse"
```

相对路径参数相对于 `source_path` 解析。因此上面的参数对应：

```text
E:\qiyu-test\test0819\prior-sparse
```

`--pose_prior` 应指向包含先验 COLMAP 模型的目录。

### 5.4 使用先验相机内参文件

```powershell
.\run-sfm.exe `
  --source_path "E:\data\scene01" `
  --prior_camera_file "camera.txt"
```

相对路径对应：

```text
E:\data\scene01\camera.txt
```

每行格式示例：

```text
CAMERA_ID, FOLD, MODEL, WIDTH, HEIGHT, PARAMS
```

## 6. 生成图像 Mask

使用以下参数启用 mask：

```text
--create_mask
--corner_width
--corner_height
--center_width
--center_height
```

四个尺寸参数不是像素值，而是图像宽度或高度的比例，取值范围为 `0~1`。

例如：

```powershell
.\run-sfm.exe `
  --source_path "E:\data\scene01" `
  --output_path "output-mask" `
  --create_mask `
  --corner_width 0.2 `
  --corner_height 0.2 `
  --center_width 0.1 `
  --center_height 0.1
```

含义：

- 每个角区域`mask`的宽度为图像宽度的 `20%`
- 每个角区域`mask`的高度为图像高度的 `20%`
- 中心区域`mask`的宽度为图像宽度的 `10%`
- 中心区域`mask`的高度为图像高度的 `10%`
- mask 中被屏蔽区域像素值为 `0`
- 其他区域像素值为 `255`

### 6.1 无子目录或只有一个子目录

程序会使用第一张图像的尺寸生成一个相机 mask：

```text
source_path\mask.png
```

并通过 COLMAP 的：

```text
--ImageReader.camera_mask_path
```

传递给特征提取命令。

这种模式适合所有图像尺寸一致的单相机数据。

### 6.2 有多个输入子目录

当 `input` 下存在多个子目录时，程序会在 `source_path\mask` 下生成与输入图像对应的 mask，并保持相对目录结构。

例如输入：

```text
scene/input/camera_1/image_0001.jpg
scene/input/camera_2/image_0001.jpg
```

生成：

```text
scene/mask/camera_1/image_0001.jpg.png
scene/mask/camera_2/image_0001.jpg.png
```

mask 文件使用 PNG 无损编码保存。对于 JPG 输入，文件名会保留原始名称并追加 `.png`，例如：

```text
image.jpg.png
```

## 7. 主要参数

### 7.1 输入、输出和设备参数

| 参数                      |           默认值 | 说明                                    |
| ------------------------- | ---------------: | --------------------------------------- |
| `--source_path`, `-s` |     `E:\debug` | 数据根目录，必须包含`input` 子目录    |
| `--output_path`, `-o` | `output-debug` | 输出目录。相对路径相对于`source_path` |
| `--no_gpu`              |             关闭 | 禁用 GPU                                |
| `--clean`               |             关闭 | 清理输出目录后再运行                    |
| `--monitor_memory`      |             关闭 | 监控 COLMAP 进程的 CPU/GPU 内存         |

### 7.2 相机参数

| 参数                              |            默认值 | 说明                             |
| --------------------------------- | ----------------: | -------------------------------- |
| `--camera`                      | `SIMPLE_RADIAL` | COLMAP 相机模型                  |
| `--camera_params`               |                空 | COLMAP 相机参数                  |
| `--default_focal_length_factor` |           `1.2` | 未读取到焦距时使用的图像尺寸比例 |
| `--prior_camera_file`, `-pcf` |                空 | 先验相机内参文件                 |
| `--init_camera`                 |              关闭 | 使用图像元数据初始化相机参数     |
| `--single_camera`, `-sc`      |             `0` | 是否所有图像共用一个相机         |
| `--single_fold`, `-sf`        |             `1` | 是否每个子目录使用一个相机       |
| `--single_image`, `-si`       |             `0` | 是否每张图像使用一个相机         |

### 7.3 特征提取参数

| 参数                                |   默认值 | 说明                                        |
| ----------------------------------- | -------: | ------------------------------------------- |
| `--feature_type`, `-ft`         | `SIFT` | `SIFT`、`ALIKED_N16ROT`、`ALIKED_N32` |
| `--max_image_size`                |   `-1` | 特征提取使用的最大图像尺寸                  |
| `--max_feature_num`, `-mfn`     | `2000` | 每张图像最多保留的特征数                    |
| `--anms_selected_num`, `-asn`   |   `-1` | ANMS 最终保留的特征数                       |
| `--cell_num`, `-cn`             |   `-1` | ANMS 网格数量                               |
| `--per_cell_num`, `-pcn`        |   `-1` | 每个网格保留的特征数                        |
| `--sift_peak_threshold`, `-spt` | `0.02` | SIFT 峰值阈值                               |
| `--sift_first_octave`, `-sfo`   |    `0` | SIFT 起始 octave                            |

### 7.4 匹配参数

| 参数                                          |         默认值 | 说明                                                                      |
| --------------------------------------------- | -------------: | ------------------------------------------------------------------------- |
| `--match_strategy`, `-ms`                 | `vocab_tree` | `exhaustive`、`sequential`、`vocab_tree`、`spatial` 或 `custom` |
| `--match_alg`, `-ma`                      | `BRUTEFORCE` | `BRUTEFORCE` 或 `LIGHTGLUE`                                           |
| `--vocab_feature_num`                       |          `0` | 词袋树检索特征数量                                                        |
| `--min_num_inliers`                         |         `15` | 有效匹配的最小内点数                                                      |
| `--min_inlier_ratio`                        |        `0.1` | 有效匹配的最小内点比例                                                    |
| `--sift_match_max_distance`, `-smmd`      |        `0.7` | SIFT 匹配距离阈值                                                         |
| `--sift_match_max_ratio`, `-smmr`         |        `0.7` | SIFT 匹配比率阈值                                                         |
| `--sequential_overlap`, `-so`             |         `15` | 顺序匹配的图像重叠数量                                                    |
| `--farest_image_distance`, `-fid`         |      `400.0` | 空间匹配的最大图像距离                                                    |
| `--max_matches_per_image`, `-mpi`         |         `50` | 每张图像最多匹配的候选图像数                                              |
| `--min_matches_per_image`, `-mni`         |         `50` | 每张图像最少匹配的候选图像数                                              |
| `--similarity_threshold`, `-st`           |       `0.75` | 相似度匹配阈值                                                            |
| `--two_view_geometry_max_error`, `-tvgme` |        `4.0` | 两视图几何最大误差                                                        |

### 7.5 建图和优化参数

| 参数                                  |     默认值 | 说明                                                                                                                                             |
| ------------------------------------- | ---------: | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--mapper`                          | `global` | `incremental`、`acc`、`global`、`hierarchical`、`hierarchical_acc`、`pos_prior`、`pose_prior_global` 或 `pose_prior_incremental` |
| `--ba_local_backend`                |  `CERES` | 局部 BA 后端：`CERES` 或 `CASPAR`                                                                                                            |
| `--ba_global_backend`               |  `CERES` | 全局 BA 后端：`CERES` 或 `CASPAR`                                                                                                            |
| `--gp_max_num_iterations`           |    `200` | 全局定位最大迭代次数                                                                                                                             |
| `--ba_ceres_max_num_iterations`     |    `200` | Ceres BA 最大迭代次数                                                                                                                            |
| `--global_mapper_min_tri_angle_deg` |    `1.0` | 全局建图最小三角角度                                                                                                                             |
| `--track_min_num_views_per_track`   |      `4` | Track 最小观测视图数                                                                                                                             |
| `--max_normalized_reproj_error`     |   `0.01` | 最大归一化重投影误差                                                                                                                             |
| `--refine_focal_length`             |      `1` | 是否优化焦距                                                                                                                                     |
| `--refine_principal_point`          |      `1` | 是否优化主点                                                                                                                                     |
| `--refine_extra_params`             |      `1` | 是否优化额外畸变参数                                                                                                                             |
| `--refine_num`                      |      `1` | 重建 refinement 次数                                                                                                                             |

### 7.6 先验位姿和后处理参数

| 参数                                    |  默认值 | 说明                                           |
| --------------------------------------- | ------: | ---------------------------------------------- |
| `--pose_prior`                        |      空 | 先验位姿或先验重建路径                         |
| `--voxel_size`                        |      空 | 先验位姿匹配的 voxel 尺寸                      |
| `--max_angle`                         | `120` | 先验位姿匹配允许的最大视角差，单位为度         |
| `--min_overlap`                       | `0.1` | 先验位姿匹配的最小视锥重叠比例                 |
| `--filt_match`                        |   `0` | 是否在建图前根据内点过滤匹配                   |
| `--filter_inlier_ratio_threshold`     | `0.2` | 匹配过滤的内点比例阈值                         |
| `--filter_inlier_num_threshold`       |  `15` | 匹配过滤的内点数阈值                           |
| `--undistort`                         |    关闭 | 建图后执行图像去畸变                           |
| `--unify_output_images`               |    关闭 | 将输出图像移动到统一目录并更新模型中的图像名称 |
| `--visualize_matches`, `-vis`       |    关闭 | 输出匹配可视化结果                             |
| `--visualize_keypoints`, `-viskpts` |    关闭 | 输出关键点可视化结果                           |

### 7.7 Mask 参数

| 参数                |  默认值 | 说明                          |
| ------------------- | ------: | ----------------------------- |
| `--create_mask`   |    关闭 | 启用 mask 自动生成            |
| `--corner_width`  | `0.0` | 角区域宽度比例，范围`0~1`   |
| `--corner_height` | `0.0` | 角区域高度比例，范围`0~1`   |
| `--center_width`  | `0.0` | 中心区域宽度比例，范围`0~1` |
| `--center_height` | `0.0` | 中心区域高度比例，范围`0~1` |

## 8. 输出目录

典型输出目录如下：

```text
output-debug/
├── distorted/
│   ├── database.db
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── images/                 # 使用 --undistort 时可能生成
├── visualization/          # 使用 --visualize_matches 时生成
├── keypoints_vis/          # 使用 --visualize_keypoints 时生成
└── run-sfm-YYYY-MM-DD-HH-MM-SS.log
```

## 9. 常见问题

### 9.1 双击程序后窗口立即关闭

建议从 PowerShell 或命令提示符执行，以便查看错误信息：

```powershell
.\run-sfm.exe --source_path "E:\data\scene01"
```

### 9.2 找不到 COLMAP 或 DLL

检查以下文件是否存在：

```text
_internal\Release-colmap-4.2.0-dev-wl-260819\colmap.exe
_internal\Release-colmap-4.2.0-dev-wl-260819\onnxruntime.dll
_internal\Release-colmap-4.2.0-dev-wl-260819\onnxruntime_providers_cuda.dll
```

发布时不要只复制 `run-sfm.exe`。

### 9.3 CUDA 相关错误

如果使用 GPU，必须保留 COLMAP 子目录中的 CUDA 和 ONNX Runtime DLL。可以先用以下方式判断是否为 GPU 相关问题：

```powershell
.\run-sfm.exe --source_path "E:\data\scene01" --no_gpu
```

如果 CPU 模式可以运行而 GPU 模式失败，应检查 NVIDIA 驱动版本以及 COLMAP 所需的 CUDA 运行库。

### 9.4 Mask 没有生效

检查：

1. 是否添加了 `--create_mask`
2. 四个比例参数是否在 `0~1` 范围内
3. `input` 下的图像是否能够读取文件头
4. 多子目录模式下，是否生成了与输入相对路径一致的 `.jpg.png` mask 文件
5. 特征提取日志中的 `ImageReader.mask_path` 或 `ImageReader.camera_mask_path` 参数是否正确

### 9.5 先验文件找不到

相对路径会拼接到 `source_path`。例如：

```powershell
--source_path "E:\data\scene01" --pose_prior "prior-sparse"
```

程序会查找：

```text
E:\data\scene01\prior-sparse
```

如文件或目录实际位于其他位置，请使用绝对路径。

## 10. 建议的生产运行命令

```powershell
.\run-sfm.exe `
  --source_path "E:\data\scene01" `
  --output_path "E:\results\scene01" `
  --feature_type SIFT `
  --match_strategy vocab_tree `
  --match_alg BRUTEFORCE `
  --mapper global `
  --create_mask `
  --corner_width 0.1 `
  --corner_height 0.1 `
  --center_width 0.2 `
  --center_height 0.2 `
  --monitor_memory
```

运行完成后，重点检查：

```text
E:\results\scene01\sparse\0\cameras.bin
E:\results\scene01\sparse\0\images.bin
E:\results\scene01\sparse\0\points3D.bin
```

以及日志文件：

```text
E:\results\scene01\run-sfm-YYYY-MM-DD-HH-MM-SS.log
```
