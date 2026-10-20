# lidar_data_parser.exe 使用说明

## 1. 程序简介

`lidar_data_parser.exe` 用于将实时采集数据转换为 COLMAP 稀疏模型。程序会依次完成：

1. 读取相机标定参数和 IMU/里程计轨迹；
2. 从 MCAP 文件中提取左右相机图像；
3. 根据里程计时间范围过滤图像；
4. 对左右鱼眼图像进行去畸变；
5. 根据里程计和外参计算 COLMAP 相机位姿；
6. 读取彩色 LAS 点云并进行两阶段下采样；
7. 估计点云法线并写出 PLY；
8. 写出 COLMAP 稀疏模型。

本文档中的程序名假设为 `lidar_data_parser.exe`。如果打包时使用了其他名称，只需将命令中的 EXE 文件名替换为实际名称。

## 2. Windows 环境要求

- Windows 10 或 Windows 11；
- 如果程序使用 GPU 相关运行库，需要安装兼容的 NVIDIA 驱动；
- EXE 发布目录中的依赖文件必须完整；
- 路径可以包含中文，但建议数据目录和输出目录使用绝对路径；
- 确保输入 LAS 文件有足够的磁盘空间和内存可供读取、采样及法线计算。

如果程序使用 PyInstaller `onedir` 模式打包，不能只复制 EXE 文件，必须复制整个发布目录。例如：

```text
lidar_data_parser/
├── lidar_data_parser.exe
├── _internal/                 # 如果打包生成该目录，必须保留
├── checkpoints/               # 如果程序打包时包含模型资源，必须保留
├── Release-colmap-*/          # 如果程序包含 COLMAP 运行库，必须保留
└── 其他 DLL 和运行文件
```

如果使用 `onefile` 模式，则按照实际打包结果保留 EXE 所需的外部资源目录。

## 3. 输入数据目录

使用 `--data_dir` 指定数据根目录。程序默认从以下位置查找输入文件：

```text
<DATA_DIR>/
├── data/
│   └── data_raw.mcap
├── info/
│   └── calibration.json
├── odom-realtime.csv
└── colorized-realtime.las
```

各文件用途如下：

| 文件                       | 用途                                                                     |
| -------------------------- | ------------------------------------------------------------------------ |
| `data/data_raw.mcap`     | 包含`/camera/left/jpeg` 和 `/camera/right/jpeg` 图像消息的 MCAP 文件 |
| `info/calibration.json`  | 左右相机内参、畸变参数及相机与激光雷达外参                               |
| `odom-realtime.csv`      | 里程计轨迹，至少包含时间戳、位置和四元数                                 |
| `colorized-realtime.las` | 彩色激光点云                                                             |

`odom-realtime.csv` 的基本列顺序为：

```text
timestamp, x, y, z, qx, qy, qz, qw, ...
```

时间戳应与 MCAP 图像的 `publish_time` 使用相同时间基准。程序默认不进行时间偏移校正，即 `--align_mode none`。

## 4. 基本用法

在 PowerShell 中执行：

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\wq-data\效果对比\82102" `
  --output_dir "E:\lidar-data\wq-data\效果对比\82102\output"
```

在 CMD 中执行时使用一行命令：

```bat
lidar_data_parser.exe --data_dir "E:\lidar-data\wq-data\效果对比\82102" --output_dir "E:\lidar-data\wq-data\效果对比\82102\output"
```

`--output_dir` 是必需参数。如果使用相对路径，相对当前命令行所在目录解析；建议直接使用绝对路径。

## 5. 推荐运行示例

### 5.1 使用默认配置

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\dataset01" `
  --output_dir "E:\lidar-data\dataset01\output"
```

当前代码默认值为：

- 点云目标数量：`500000`；
- 第二阶段体素边长：`0.5`；
- 第一阶段随机采样比例：`0.6`；
- 时间对齐模式：`none`；
- 输出格式：`txt`；
- 去畸变插值：`cubic`。

### 5.2 调整点云下采样参数

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\dataset01" `
  --output_dir "E:\lidar-data\dataset01\output-300k" `
  --num_points 300000 `
  --voxel_size 0.5 `
  --random_ratio 0.6
```

参数含义：

- `--num_points`：最终点云目标数量；
- `--voxel_size`：第二阶段体素边长，单位与 LAS 坐标一致；
- `--random_ratio`：第一阶段均匀随机采样所占比例，取值通常为 `0~1`。

两阶段采样流程为：

1. 从原始点云中随机抽取 `num_points * random_ratio` 个点；
2. 删除第一阶段已选点；
3. 对剩余点执行按体素随机采样；
4. 如果结果超过目标数量，再随机裁剪到 `num_points`。

### 5.3 使用制造商提供的里程计文件

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\dataset01" `
  --output_dir "E:\lidar-data\dataset01\output-refined" `
  --odom_path "E:\lidar-data\dataset01\odom-refined.csv"
```

### 5.4 使用外部 MCAP 文件

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\dataset01" `
  --output_dir "E:\lidar-data\dataset01\output" `
  --mcap_path "E:\recordings\data_raw.mcap"
```

### 5.5 选择坐标轴对齐方式

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\dataset01" `
  --output_dir "E:\lidar-data\dataset01\output" `
  --axis_align x180
```

可选值：

- `none`：不进行轴向修正，默认值；
- `x180`：绕 X 轴旋转 180 度；
- `y180`：绕 Y 轴旋转 180 度；
- `z180`：绕 Z 轴旋转 180 度。

只有在确认设备坐标系与 COLMAP 坐标系存在轴向差异时才修改该参数。

### 5.7 跳过已经完成的步骤

如果输出目录中已经有对应的中间数据，可以跳过部分步骤：

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\dataset01" `
  --output_dir "E:\lidar-data\dataset01\output" `
  --skip_extract `
  --skip_undistort `
  --skip_pointcloud
```

参数说明：

- `--skip_extract`：不再从 MCAP 提取图像，使用已有的 `fisheye-images/left` 和 `fisheye-images/right`；
- `--skip_undistort`：不再去畸变，使用已有的去畸变图像；
- `--skip_pointcloud`：跳过 LAS 点云处理，不生成新的 `points3D.ply`。

跳过步骤前必须确认相关输入文件已经存在，否则程序可能报错或生成不完整的 COLMAP 模型。

## 6. 完整参数表

| 参数                   | 默认值                            | 说明                                                         |
| ---------------------- | --------------------------------- | ------------------------------------------------------------ |
| `--data_dir`         | 内置默认路径                      | 输入数据根目录                                               |
| `--output_dir`       | 必需                              | 输出目录                                                     |
| `--align_mode`       | `none`                          | 时间对齐方式：`mean`、`start`、`end`、`none`         |
| `--odom_path`        | `<data_dir>/odom-realtime.csv`  | 自定义里程计 CSV 路径                                        |
| `--mcap_path`        | `<data_dir>/data/data_raw.mcap` | 自定义 MCAP 路径                                             |
| `--axis_align`       | `none`                          | 坐标轴对齐：`none`、`x180`、`y180`、`z180`           |
| `--num_points`       | `500000`                        | 点云目标数量                                                 |
| `--voxel_size`       | `0.5`                           | 体素边长                                                     |
| `--random_ratio`     | `0.6`                           | 第一阶段随机采样比例                                         |
| `--fmt`              | `txt`                           | COLMAP 输出格式：`txt`、`bin`、`both`                  |
| `--max_workers`      | `0`                             | 去畸变线程数；`0` 表示使用 CPU 逻辑核心数的约 60%          |
| `--undistort_interp` | `cubic`                         | 去畸变插值：`nearest`、`linear`、`cubic`、`lanczos4` |
| `--skip_extract`     | 关闭                              | 跳过 MCAP 图像提取                                           |
| `--skip_undistort`   | 关闭                              | 跳过去畸变                                                   |
| `--skip_pointcloud`  | 关闭                              | 跳过点云处理                                                 |

## 7. 输出目录结构

一次完整运行后，输出目录通常包含：

```text
output/
├── fisheye-images/
│   ├── left/
│   └── right/
├── undistorted-images/
│   ├── left/
│   └── right/
└── lidar-sparse/
    ├── cameras.txt / cameras.bin
    ├── images.txt / images.bin
    ├── points3D.txt / points3D.bin
    └── points3D.ply
```

说明：

- `fisheye-images/`：从 MCAP 提取的原始鱼眼图像；
- `undistorted-images/`：去畸变后的左右图像；
- `lidar-sparse/`：COLMAP 稀疏模型及点云结果；
- `points3D.ply`：下采样并计算法线后的彩色点云；
- `cameras.*`：相机内参；
- `images.*`：图像位姿及图像名称；
- `points3D.*`：COLMAP 点云模型。

`--fmt txt` 只生成文本模型，`--fmt bin` 只生成二进制模型，`--fmt both` 同时生成两种格式。

## 8. 运行日志

程序会在控制台输出各阶段状态和耗时，包括：

- 标定和里程计读取；
- MCAP 图像提取；
- 图像时间范围过滤；
- 图像去畸变；
- 相机位姿插值；
- 点云处理；
- COLMAP 模型写出。

点云下采样阶段会输出体素数量和每个体素的采样数量，例如：

```text
[voxel_random_sample] voxel_size=0.5, voxel_count=..., points_per_voxel=...
```

## 9. 常见问题

### 9.1 提示找不到图像通道

确认 MCAP 中存在以下主题：

```text
/camera/left/jpeg
/camera/right/jpeg
```

同时确认消息类型是支持的压缩图像类型，并且 MCAP 的消息编码为 ROS1 或 ROS2 CDR。

### 9.2 提示找不到输入文件

检查以下路径是否存在：

```text
<data_dir>\data\data_raw.mcap
<data_dir>\info\calibration.json
<data_dir>\odom-realtime.csv
<data_dir>\colorized-realtime.las
```

如果文件位置不同，使用 `--mcap_path` 或 `--odom_path` 指定实际路径。当前版本没有为 LAS 文件提供单独的命令行路径参数，因此 LAS 文件需要放在 `<data_dir>\colorized-realtime.las`。

### 9.3 内存不足或运行很慢

可以采取以下措施：

- 降低 `--num_points`，例如从 `500000` 改为 `200000`；
- 增大 `--voxel_size`，减少体素数量；
- 降低 `--max_workers`，减少去畸变阶段的并发内存占用；
- 使用 `--skip_extract`、`--skip_undistort` 或 `--skip_pointcloud` 避免重复处理；
- 确保输出目录所在磁盘有足够空间。

### 9.4 去畸变速度较慢

可以尝试：

```powershell
--undistort_interp linear
```

`linear` 通常比 `cubic` 和 `lanczos4` 更快。若 CPU 内存充足，可以适当增加 `--max_workers`，但线程过多可能导致内存压力或磁盘 I/O 竞争。

### 9.5 图像被过滤掉

程序会删除落在里程计时间范围之外的图像。如果左右相机全部图像都被过滤，检查：

1. MCAP 图像文件名使用的时间字段；
2. `odom-realtime.csv` 的时间单位和时间基准；
3. 是否需要使用 `--align_mode mean`、`start` 或 `end`；
4. 图像时间戳是否与里程计时间戳同为纳秒。

### 9.6 输出 COLMAP 模型方向不正确

尝试确认设备坐标系约定后使用：

```powershell
--axis_align x180
```

不要在未确认坐标系的情况下盲目切换轴向参数，应结合相机位姿和点云位置检查结果。

## 10. 建议的标准命令

```powershell
.\lidar_data_parser.exe `
  --data_dir "E:\lidar-data\wq-data\效果对比\82102" `
  --output_dir "E:\lidar-data\wq-data\效果对比\82102\output" `
  --num_points 500000 `
  --voxel_size 0.5 `
  --random_ratio 0.6 `
  --fmt both `
  --undistort_interp linear
```

运行完成后，主要检查：

```text
<output_dir>\lidar-sparse\cameras.txt
<output_dir>\lidar-sparse\images.txt
<output_dir>\lidar-sparse\points3D.ply
```

这些文件存在且控制台最后出现 `COLMAP model ready` 时，通常表示处理完成。
