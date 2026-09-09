# run-sfm.exe 使用说明

## 1. 程序简介

`run-sfm.exe` 用于根据图像、相机参数和可选的位姿先验生成 COLMAP 稀疏模型。

程序发布时请保留完整的 `run-sfm` 文件夹，不能只复制 `run-sfm.exe`。

```text
run-sfm/
├── run-sfm.exe
├── _internal/
├── checkpoints/
└── Release-colmap-*/
```

## 2. 通用说明

在 PowerShell 中运行程序时，路径建议使用绝对路径，并使用引号包裹路径：

```powershell
.\run-sfm.exe -s "数据源目录" -o "输出目录"
```

其中：

- `-s`：输入数据源目录；
- `-o`：输出目录，可选。未指定时使用默认输出目录与 -s 指定的数据源目录相同。

程序运行完成后，COLMAP 结果会保存在输出目录下。

## 3. 数据源一：手机拍摄图像

### 3.1 输入目录

手机图像放在数据源目录的 `input` 子目录中：

```text
手机数据/
└── input/
    ├── image_0001.jpg
    ├── image_0002.jpg
    ├── image_0003.jpg
    └── ...
```

如果图像分布在多个子目录中，也可以保留原有目录结构：

```text
手机数据/
└── input/
    ├── camera_1/
    │   ├── image_0001.jpg
    │   └── image_0002.jpg
    └── camera_2/
        ├── image_0001.jpg
        └── image_0002.jpg
```

### 3.2 使用方法

手机拍摄图像只需要指定 `-s` 参数即可：

```powershell
.\run-sfm.exe `
  -s "E:\data\phone_scene"
```

如果需要指定输出目录：

```powershell
.\run-sfm.exe `
  -s "E:\data\phone_scene" `
  -o "E:\data\phone_scene\output"
```

## 4. 数据源二：手机 + 无人机图像

手机和无人机图像的使用方法与单纯手机拍摄图像相同，不需要额外指定特殊参数。

### 4.1 输入目录示例

将手机和无人机图像统一放在 `input` 目录下：

```text
手机无人机数据/
└── input/
    ├── phone/
    │   ├── phone_0001.jpg
    │   └── phone_0002.jpg
    └── drone/
        ├── drone_0001.jpg
        └── drone_0002.jpg
```

### 4.2 使用方法

只需要指定 `-s` 参数：

```powershell
.\run-sfm.exe `
  -s "E:\data\phone_drone_scene"
```

也可以同时指定输出目录：

```powershell
.\run-sfm.exe `
  -s "E:\data\phone_drone_scene" `
  -o "E:\data\phone_drone_scene\output"
```

手机和无人机图像应尽量使用清晰、连续且具有足够重叠区域的图像。多个子目录会作为输入图像目录结构保留。

## 5. 数据源三：激光图像 + 激光先验位姿和内参

激光数据需要同时提供：

- 激光图像；
- 激光先验位姿；
- 激光相机内参。

以上数据需要在激光扫描仪原始数据上执行 “lidar_data_paser.exe” 程序后输出

### 5.1 输入目录

推荐目录结构如下：

```text
激光数据/
├── undistored-images/
│   ├── left/
│   │   ├── image_0001.jpg
│   │   └── image_0002.jpg
│   └── right/
│       ├── image_0001.jpg
│       └── image_0002.jpg
├── lidar-sparse/
│   ├── cameras.bin 或 cameras.txt
│   ├── images.bin 或 images.txt
│   └── points3D.bin 或 points3D.txt
```

`--pose_prior` 应指向包含激光先验位姿及相机内参的 COLMAP 模型目录，例如 `lidar-sparse`。

### 5.2 使用方法

激光数据需要指定以下参数：

- `-s`：激光数据源目录；
- `--image_dir`：图像所在目录，执行lidar_data_parse.exe后产生的 `undistorted_images`；
- `--pose_prior`：激光先验位姿目录, 执行lidar_data_parse.exe后产生的 `lidar-sparse` ；
- `-ft SIFT`：使用 SIFT 特征；
- `-ma SIFT_BRUTEFORCE`：使用 SIFT 暴力匹配；
- `-mapper global：`使用 global 模式稀疏建图
- `--cropped_width 2000 ：` 去畸变之后将图像宽度裁切到2000；
- `--cropped_height 2000: `去畸变之后将图像高度裁切到2000.

示例：

```powershell
.\run-sfm.exe `
  -s "E:\data\laser_scene" `
  --image_dir "undistorted-images" `
  --pose_prior "lidar-sparse" `
  -ft SIFT `
  -ma SIFT_BRUTEFORCE `
  --mapper global `
  --cropped_width 2000 `
  --cropped_height 2000 `
```

也可以指定输出目录：

```powershell
.\run-sfm.exe `
  -s "E:\data\laser_scene" `
  --image_dir "input" `
  --pose_prior "prior-sparse" `
  -ft SIFT `
  -ma SIFT_BRUTEFORCE `
  -o "E:\data\laser_scene\output"
```

## 6. 三类数据源命令汇总

### 手机拍摄图像

```powershell
.\run-sfm.exe -s "E:\data\phone_scene"
```

### 手机 + 无人机图像

```powershell
.\run-sfm.exe -s "E:\data\phone_drone_scene"
```

### 激光图像 + 先验位姿和内参

```powershell
.\run-sfm.exe `
  -s "E:\data\laser_scene" `
  --image_dir "input" `
  --pose_prior "prior-sparse" `
  -ft SIFT `
  -ma SIFT_BRUTEFORCE
  --mapper global `
  --cropped_width 2000 `
  --cropped_height 2000 `
```

## 7. 常见注意事项

1. `-s` 指定的是数据源根目录，不是单张图像所在目录。
2. 手机和无人机图像应放在 `input` 或 `--image_dir` 指定的目录下。
3. 激光数据必须保证 `--pose_prior` 指向有效的 COLMAP 先验模型目录。
4. 相对路径通常相对于 `-s` 指定的数据源目录解析。
5. 如果命令行提示参数无效，请先执行：

```powershell
.\run-sfm.exe --help
```
