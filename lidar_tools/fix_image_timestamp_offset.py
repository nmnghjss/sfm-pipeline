#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_image_timestamp_offset.py
================================
修正图像文件名中的时间戳偏置，使其与 ImgPose.txt 的记录对齐，并把结果写到另一个目录。

背景
----
* ``images_dir`` 下的图像文件名本身就是采集时间戳（如 ``1787294284191525888.jpg``）；
* ``ImgPose.txt`` 每行形如 ``left/1787294284191525888.jpg x y z roll pitch yaw qx qy qz qw timestamp``；
* 图像名时间戳与 ImgPose.txt 记录的时间戳之间存在**系统偏差**（相机时钟 / 取整等），
  同一帧在两边的时间戳相差一个近似常量；
* 且 images_dir 中的图像并非每一张都在 ImgPose.txt 中有记录。

做法
----
1. 解析 ImgPose.txt（第一列图像相对路径，时间戳取表头 ``timestamp`` 列 / 最后一列 /
   文件名内嵌数字），按第一层目录（left、right…）分组；
2. 解析图像文件名中的时间戳（默认自动识别 s / ms / us / ns，亦可用 ``--img_ts_unit`` 指定）；
3. 估计偏置 Δ（= 图像时间戳 − 位姿时间戳）时会同时尝试多种假设，用
   「命中数 → 命中残差中位数 → |offset|」打分取最优，最后用残差中位数迭代精化：
   * histogram：最近邻差值直方图的众数（``--bin_ms`` 分箱），适合偏置小、两组时间范围重叠的常见情形；
   * rank0.00/0.25/0.50/0.75/1.00：按两条序列的分位数对齐，适合偏置很大（甚至不同时间纪元）时。
   也可用 ``--offset_ns`` 直接指定，跳过估计；
4. 逐张图像 ``new_ts = img_ts - Δ``：
   * 若与最近的位姿时间戳相差 ``<= --match_tol`` → **命中**，直接用位姿文件里的文件名；
   * 否则 → **未命中**，用 ``new_ts`` 替换文件名中的时间戳数字；
5. 按 ``--mode`` 复制 / 硬链接 / 移动到 ``--output_dir``（保持相对目录结构），
   并输出 ``name_mapping.csv`` 与 ``offset_report.json``。

示例
----
    python fix_image_timestamp_offset.py \
        --images_dir "E:\\lidar-data\\wq-data\\效果对比\\82101\\桌面端处理后\\undistort" \
        --pose_file  "E:\\lidar-data\\wq-data\\效果对比\\82101\\桌面端处理后\\ImgPose.txt" \
        --output_dir "E:\\lidar-data\\wq-data\\效果对比\\82101\\桌面端处理后\\undistort-fixed"
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_HEADER_HINTS = {"index", "image", "image_path", "file_path", "name", "filename"}
_UNIT_SCALE = {"s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3, "ns": 1}
_DIGITS = re.compile(r"\d{6,}")


# ---------------------------------------------------------------------------
# 时间戳解析
# ---------------------------------------------------------------------------
def to_ns(text: str, unit: str = "auto") -> int:
    """把时间戳文本转成纳秒整数；unit='auto' 时按数量级猜测 s/ms/us/ns。"""
    s = text.strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty timestamp")
    if "." in s or "e" in s.lower():
        value = Decimal(s) * (10 ** 9 if unit == "auto" else _UNIT_SCALE[unit])
        return int(value.to_integral_value())
    value = int(Decimal(s))
    if unit != "auto":
        return value * _UNIT_SCALE[unit]
    mag = abs(value)
    if mag >= 10 ** 17:          # 纳秒（约 2026 年为 1.78e18）
        return value
    if mag >= 10 ** 14:          # 微秒
        return value * 10 ** 3
    if mag >= 10 ** 11:          # 毫秒
        return value * 10 ** 6
    return value * 10 ** 9       # 秒


def split_stem(stem: str, unit: str):
    """从文件名主干中取出时间戳：返回 (ts_ns, prefix, suffix)，取不到返回 None。

    取**最后一组**连续数字（>=6 位），前后缀原样保留，便于把新时间戳写回原位置。
    """
    last = None
    for last in _DIGITS.finditer(stem):
        pass
    if last is None:
        return None
    return to_ns(last.group(0), unit), stem[:last.start()], stem[last.end():]


def _group_of(name: str) -> str:
    """位姿/图像相对路径的第一层目录，作为相机分组（无子目录则为空串）。"""
    normalized = name.replace("\\", "/")
    return normalized.split("/", 1)[0] if "/" in normalized else ""


def _nearest(sorted_values, target) -> int:
    """返回排序列表中距离 target 最近的元素下标。"""
    index = bisect.bisect_left(sorted_values, target)
    if index <= 0:
        return 0
    if index >= len(sorted_values):
        return len(sorted_values) - 1
    before, after = sorted_values[index - 1], sorted_values[index]
    return index - 1 if (target - before) <= (after - target) else index


def parse_pose_file(path: Path, ts_col: str | None, ts_unit: str):
    """解析 ImgPose.txt，返回 [(name, ts_ns), ...]。

    ts_col: 列名（表头命中）或 1 开始的列号；None 时优先找表头中含 time/stamp 的列，
            否则取最后一列；该列解析失败时回退到文件名内嵌数字。
    """
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_lines:
        raise ValueError(f"空文件: {path}")

    header = None
    first = data_lines[0].replace(",", " ").split()
    try:
        to_ns(first[-1], ts_unit)
        is_header = first[0].lower() in _HEADER_HINTS and len(first) > 11
    except (ValueError, InvalidOperation):
        is_header = True
    if is_header:
        header = [f.strip().lower() for f in first]
        data_lines = data_lines[1:]

    col = None
    if ts_col:
        if ts_col.lstrip("-").isdigit():
            col = int(ts_col) - 1
        elif header and ts_col.strip().lower() in header:
            col = header.index(ts_col.strip().lower())
        else:
            raise ValueError(f"{path}: 找不到时间戳列 {ts_col!r}（表头: {header}）")
    elif header:
        for keyword in ("timestamp", "time", "stamp"):
            for i, name in enumerate(header):
                if keyword in name:
                    col = i
                    break
            if col is not None:
                break
    if col is None:
        col = -1  # 最后一列

    poses = []
    for lineno, line in enumerate(data_lines, 1):
        fields = line.replace(",", " ").split()
        if not fields:
            continue
        name = fields[0].replace("\\", "/")
        ts = None
        if -len(fields) <= col < len(fields):
            try:
                ts = to_ns(fields[col], ts_unit)
            except (ValueError, InvalidOperation):
                ts = None
        if ts is None:  # 回退：从文件名里取
            found = _DIGITS.findall(name)
            if not found:
                print(f"[warn] {path}: 第 {lineno} 行无法解析时间戳，已跳过: {line[:80]}")
                continue
            ts = to_ns(found[-1], ts_unit)
        poses.append((name, ts))
    if not poses:
        raise ValueError(f"{path}: 未解析到任何位姿")
    return poses


# ---------------------------------------------------------------------------
# 偏置估计
# ---------------------------------------------------------------------------
def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _score_offset(img_ts_sorted, pose_ts_sorted, offset, tol_ns: int):
    """给定偏置下，落在最近位姿容差内的图像数。"""
    matched = 0
    residuals = []
    for ts in img_ts_sorted:
        shifted = ts - offset
        index = _nearest(pose_ts_sorted, shifted)
        residual = shifted - pose_ts_sorted[index]
        residuals.append(residual)
        if abs(residual) <= tol_ns:
            matched += 1
    return matched, residuals


def _refine_offset(img_ts_sorted, pose_ts_sorted, offset, tol_ns: int, iterations: int = 20):
    """由初值出发迭代对齐：用残差中位数逐步校正 offset，返回 (offset, 命中数, 残差列表)。"""
    offset = int(offset)
    for _ in range(iterations):
        _, residuals = _score_offset(img_ts_sorted, pose_ts_sorted, offset, tol_ns)
        step = int(round(_median(residuals)))
        if step == 0:
            break
        offset += step
    matched, residuals = _score_offset(img_ts_sorted, pose_ts_sorted, offset, tol_ns)
    return offset, matched, residuals


def estimate_offset(img_ts_list, pose_ts_sorted, bin_ns: int, min_samples: int, tol_ns: int):
    """估计 offset = 图像时间戳 − 位姿时间戳。

    同时尝试多种假设并用「命中数 → 命中残差中位数 → |offset|」打分取最优：
      * histogram —— 最近邻差值直方图众数：适用于偏置较小、两组时间范围重叠的常见情形；
      * rank0.00/0.25/0.50/0.75/1.00 —— 按两条序列的分位数对齐：适用于偏置很大
        （甚至完全不同的时间纪元）且两组覆盖同一时间跨度时。

    返回 (offset_ns or None, info)。
    """
    img_sorted = sorted(img_ts_list)
    if not img_sorted or not pose_ts_sorted:
        return None, dict(candidates=0, reason="空输入")

    attempts = []
    candidates = [ts - pose_ts_sorted[_nearest(pose_ts_sorted, ts)] for ts in img_sorted]
    if len(candidates) >= min_samples:
        bins = Counter((c + bin_ns // 2) // bin_ns for c in candidates)
        best_bin, hits = bins.most_common(1)[0]
        center = best_bin * bin_ns
        near = [c for c in candidates if abs(c - center) <= 5 * bin_ns]
        attempts.append(("histogram", int(round(sum(near) / len(near))), hits))

    n, m = len(img_sorted), len(pose_ts_sorted)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        offset = img_sorted[min(n - 1, int(frac * (n - 1)))] - pose_ts_sorted[min(m - 1, int(frac * (m - 1)))]
        attempts.append((f"rank{frac:.2f}", offset, 0))

    results = []
    best = None
    for label, offset0, hits in attempts:
        offset, matched, residuals = _refine_offset(img_sorted, pose_ts_sorted, offset0, tol_ns)
        ok = sorted(abs(r) for r in residuals if abs(r) <= tol_ns)
        median_abs = _median(ok) if ok else float("inf")
        results.append(dict(method=label, start_ns=int(offset0), offset_ns=offset,
                            matched=matched, bin_hits=hits,
                            median_abs_residual_ns=median_abs,
                            max_abs_residual_ns=(ok[-1] if ok else None)))
        # 打分：命中数优先；同分时看残差中位数（真对齐通常贴合得更紧，可排除“整帧错位”的
        # 低残差假象）；再同分则取 |offset| 更小者（假定时钟偏差本身不大）。
        score = (matched, -median_abs, -abs(offset))
        if best is None or score > best[0]:
            best = (score, offset, label, matched)

    offset, matched, label = best[1], best[3], best[2]
    info = dict(candidates=len(candidates), method=label, matched=matched,
                images=len(img_sorted), poses=len(pose_ts_sorted), attempts=results)
    return offset, info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True, type=Path, help="待修正的图像目录（递归扫描）")
    ap.add_argument("--pose_file", type=Path, default=None,
                    help="ImgPose.txt 路径（默认 <input_dir>/ImgPose.txt）")
    ap.add_argument("--input_dir", type=Path, default=None,
                    help="仅用于推断 pose_file 默认位置（默认 images_dir 的父目录）")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="输出目录（默认 <images_dir>-fixed）")
    ap.add_argument("--mode", choices=("copy", "hardlink", "move"), default="copy",
                    help="输出方式（默认 copy；hardlink 失败会自动退化为 copy）")
    ap.add_argument("--extensions", default=",".join(DEFAULT_EXTS),
                    help="图像扩展名过滤（逗号分隔）")
    ap.add_argument("--img_ts_unit", choices=("auto", "s", "ms", "us", "ns"), default="auto",
                    help="图像名时间戳单位（默认 auto）")
    ap.add_argument("--pose_ts_unit", choices=("auto", "s", "ms", "us", "ns"), default="auto",
                    help="ImgPose.txt 时间戳单位（默认 auto）")
    ap.add_argument("--pose_ts_col", default=None,
                    help="时间戳列：列名（表头）或 1 开始的列号（默认自动识别）")
    ap.add_argument("--offset_ns", type=int, default=None,
                    help="手动指定偏置 = 图像时间戳 − 位姿时间戳（跳过自动估计）")
    ap.add_argument("--match_tol", type=float, default=0.02,
                    help="命中容差，秒（默认 0.02）")
    ap.add_argument("--bin_ms", type=float, default=10.0,
                    help="偏置直方图分箱宽度，毫秒（默认 10）")
    ap.add_argument("--min_samples", type=int, default=3,
                    help="每组估计偏置所需的最小样本数（默认 3）")
    ap.add_argument("--skip_unmatched", action="store_true",
                    help="丢弃在 ImgPose.txt 中没有记录的图像")
    ap.add_argument("--unmatched_subdir", default=None,
                    help="把未命中的图像输出到该子目录（默认与原结构相同）")
    ap.add_argument("--allow_duplicates", action="store_true",
                    help="输出名冲突时跳过后出现的文件（默认报错退出）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖输出目录中已存在的文件")
    ap.add_argument("--dry_run", action="store_true", help="只打印计划并写报告，不落盘图像")
    ap.add_argument("--report_csv", type=Path, default=None,
                    help="映射表输出路径（默认 <output_dir>/name_mapping.csv）")
    ap.add_argument("--report_json", type=Path, default=None,
                    help="报告输出路径（默认 <output_dir>/offset_report.json）")
    args = ap.parse_args()

    images_dir = args.images_dir.resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"[error] images_dir 不存在: {images_dir}")
    pose_file = (args.pose_file or ((args.input_dir or images_dir.parent) / "ImgPose.txt")).resolve()
    if not pose_file.is_file():
        raise SystemExit(f"[error] 找不到 ImgPose.txt: {pose_file}")
    output_dir = (args.output_dir or images_dir.with_name(images_dir.name + "-fixed")).resolve()
    report_csv = (args.report_csv or (output_dir / "name_mapping.csv")).resolve()
    report_json = (args.report_json or (output_dir / "offset_report.json")).resolve()

    exts = {("." + e.strip().lstrip(".").lower()) for e in args.extensions.split(",") if e.strip()}
    image_paths = [p for p in sorted(images_dir.rglob("*"))
                   if p.is_file() and p.suffix.lower() in exts]
    if not image_paths:
        raise SystemExit(f"[error] {images_dir} 下没有匹配 {sorted(exts)} 的图像")

    # ---- 1. 位姿 ----------------------------------------------------------
    poses = parse_pose_file(pose_file, args.pose_ts_col, args.pose_ts_unit)
    pose_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name, ts in poses:
        pose_groups[_group_of(name)].append((ts, name))
    for key in pose_groups:
        pose_groups[key].sort(key=lambda item: item[0])
    global_poses = sorted((ts, name) for name, ts in poses)
    print(f"[pose] {pose_file.name}: {len(poses)} 条记录, 分组 "
          f"{ {k or '<flat>': len(v) for k, v in pose_groups.items()} }")

    # ---- 2. 图像 ----------------------------------------------------------
    entries = []
    unparsed = []
    for path in image_paths:
        rel = path.relative_to(images_dir)
        stem_info = split_stem(path.stem, args.img_ts_unit)
        if stem_info is None:
            unparsed.append(rel.as_posix())
            continue
        ts, prefix, suffix = stem_info
        group = _group_of(rel.as_posix())
        entries.append(dict(path=path, rel=rel, ts=ts, prefix=prefix, suffix=suffix,
                            ext=path.suffix, group=group,
                            key=group if group in pose_groups else ""))
    if unparsed:
        print(f"[warn] {len(unparsed)} 个文件名中未找到时间戳，已跳过: {unparsed[:5]}")
    if not entries:
        raise SystemExit("[error] 没有解析出任何带时间戳的图像文件名")

    # ---- 3. 偏置估计（按相机分组） ----------------------------------------
    bin_ns = max(1, int(round(args.bin_ms * 1e6)))
    match_tol_ns = int(round(args.match_tol * 1e9))
    by_key: dict[str, list[int]] = defaultdict(list)
    for item in entries:
        by_key[item["key"]].append(item["ts"])

    global_pose_list = [ts for ts, _ in global_poses]
    global_offset = None
    if args.offset_ns is None:
        global_offset, _ = estimate_offset(
            [item["ts"] for item in entries], global_pose_list,
            bin_ns, args.min_samples, match_tol_ns)
        if global_offset is None:
            raise SystemExit("[error] 无法估计时间戳偏置，请用 --offset_ns 手动指定")

    offsets: dict[str, int] = {}
    offset_info: dict[str, dict] = {}
    for key, ts_list in sorted(by_key.items()):
        pose_list = pose_groups.get(key) or global_poses
        pose_ts = [ts for ts, _ in pose_list]
        if args.offset_ns is not None:
            offsets[key] = args.offset_ns
            offset_info[key] = dict(source="manual", offset_ns=args.offset_ns)
            continue
        offset, info = estimate_offset(ts_list, pose_ts, bin_ns, args.min_samples, match_tol_ns)
        if offset is None:
            offset, info = global_offset, dict(info, method="global", matched=0)
            info["source"] = "global-fallback"
        else:
            info["source"] = "estimated"
        offsets[key] = offset
        info["offset_ns"] = offset
        offset_info[key] = info

    for key in sorted(offsets):
        off = offsets[key]
        info = offset_info[key]
        matched = info.get("matched")
        extra = (f"命中 {matched}/{info.get('images')}（{info.get('method')}）"
                 if matched is not None else info.get("source"))
        print(f"[offset] group={key or '<flat>'}: {off} ns "
              f"({off / 1e9:.6f} s, 样本 {len(by_key[key])}, {extra})")

    # ---- 4. 计算新文件名 --------------------------------------------------
    plan = []
    for item in entries:
        key = item["key"]
        pose_list = pose_groups.get(key) or global_poses
        pose_ts = [ts for ts, _ in pose_list]
        pose_names = [name for _, name in pose_list]
        offset = offsets[key]
        new_ts = item["ts"] - offset
        index = _nearest(pose_ts, new_ts)
        residual = new_ts - pose_ts[index]
        matched = abs(residual) <= match_tol_ns
        if matched:
            new_name = Path(pose_names[index]).name
        else:
            new_name = f"{item['prefix']}{new_ts}{item['suffix']}{item['ext']}"
        if matched:
            out_rel = item["rel"].parent / new_name
        elif args.unmatched_subdir:
            out_rel = Path(args.unmatched_subdir) / item["rel"].parent / new_name
        else:
            out_rel = item["rel"].parent / new_name
        plan.append(dict(item, new_ts=new_ts, new_name=new_name, matched=matched,
                         residual_ns=residual, out_rel=out_rel, pose_name=pose_names[index]))

    matched_n = sum(1 for p in plan if p["matched"])
    print(f"[match] 命中 {matched_n}/{len(plan)}（容差 {match_tol_ns / 1e6:.1f} ms）")
    if matched_n:
        worst = max(abs(p["residual_ns"]) for p in plan if p["matched"])
        print(f"[match] 命中样本最大残差 {worst} ns")
    if matched_n / len(plan) < 0.5:
        print("[warn] 命中率偏低：偏置可能不是常量，或时间戳单位/位姿文件列不对；"
              "可用 --img_ts_unit/--pose_ts_unit/--offset_ns 排查")

    # ---- 5. 冲突检查 ------------------------------------------------------
    seen: dict[Path, Path] = {}
    collisions = []
    for item in plan:
        key_rel = item["out_rel"]
        if key_rel in seen:
            collisions.append((seen[key_rel], item["rel"], key_rel))
        else:
            seen[key_rel] = item["rel"]
    if collisions and not args.allow_duplicates:
        detail = "; ".join(f"{a} & {b} -> {c}" for a, b, c in collisions[:5])
        raise SystemExit(f"[error] 输出文件名冲突（{len(collisions)} 处），偏置可能有误: {detail}")

    # ---- 6. 落盘 ----------------------------------------------------------
    written = skipped = dup_skipped = 0
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        if not item["matched"] and args.skip_unmatched:
            continue
        if args.allow_duplicates and item["out_rel"] not in seen:
            dup_skipped += 1
            continue
        dst = output_dir / item["out_rel"]
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            written += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "move":
            shutil.move(str(item["path"]), str(dst))
        elif args.mode == "hardlink":
            try:
                os.link(str(item["path"]), str(dst))
            except OSError:
                shutil.copy2(str(item["path"]), str(dst))
        else:
            shutil.copy2(str(item["path"]), str(dst))
        written += 1

    # ---- 7. 报告 ----------------------------------------------------------
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["old_rel_path", "new_rel_path", "old_ts_ns", "new_ts_ns",
                         "matched", "residual_ns", "pose_name"])
        for item in plan:
            writer.writerow([item["rel"].as_posix(), item["out_rel"].as_posix(),
                             item["ts"], item["new_ts"], int(item["matched"]),
                             item["residual_ns"], item["pose_name"]])
    summary = dict(
        images_dir=str(images_dir), pose_file=str(pose_file),
        output_dir=str(output_dir), mode=args.mode, dry_run=args.dry_run,
        images=len(plan), unparsed=unparsed, matched=matched_n,
        unmatched=len(plan) - matched_n,
        match_tol_ns=match_tol_ns,
        offsets={k or "<flat>": dict(offset_info[k], seconds=v / 1e9)
                 for k, v in offsets.items()},
        unmatched_images=[dict(old=p["rel"].as_posix(), new=p["out_rel"].as_posix(),
                               new_ts_ns=p["new_ts"], nearest_pose=p["pose_name"],
                               residual_ns=p["residual_ns"])
                          for p in plan if not p["matched"]],
        collisions=[dict(a=a.as_posix(), b=b.as_posix(), target=c.as_posix())
                    for a, b, c in collisions],
    )
    with report_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    verb = "计划写出" if args.dry_run else "已写出"
    print(f"[done] {verb} {written} 个文件 -> {output_dir}（跳过已存在 {skipped}）")
    print(f"[done] 映射表: {report_csv}")
    print(f"[done] 报告:   {report_json}")


if __name__ == "__main__":
    main()
