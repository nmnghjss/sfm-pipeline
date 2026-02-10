import pandas as pd
import matplotlib.pyplot as plt

# CSV 文件路径（改成你的实际路径）
csv_path = "videos_extracted_num.csv"

# 读取 CSV
df = pd.read_csv(csv_path)

# 按 fixed_extracted 从小到大排序
df = df.sort_values(by="fix_extracted_num")

# 提取列
x = df["video_name"]
y_auto = df["auto_extracted_num"]
y_fixed = df["fix_extracted_num"]

# 计算平均值
auto_mean = y_auto.mean()
fixed_mean = y_fixed.mean()

# 创建图像
plt.figure(figsize=(12, 6))
# 先画曲线，并接收返回对象
line_auto, = plt.plot(x, y_auto, marker="o", label="auto_extracted")
line_fixed, = plt.plot(x, y_fixed, marker="s", label="fix_extracted")

# 获取曲线颜色
auto_color = line_auto.get_color()
fixed_color = line_fixed.get_color()

# 用同样颜色画平均线
plt.axhline(auto_mean,
            color=auto_color,
            linestyle="--",
            linewidth=2,
            label=f"auto_mean = {auto_mean:.2f}")

plt.axhline(fixed_mean,
            color=fixed_color,
            linestyle="--",
            linewidth=2,
            label=f"fix_mean = {fixed_mean:.2f}")

plt.xticks(rotation=60, ha="right")

plt.xlabel("video_name")
plt.ylabel("value")
plt.title("auto_extracted vs fix_extracted")

plt.legend()
plt.tight_layout()
plt.show()