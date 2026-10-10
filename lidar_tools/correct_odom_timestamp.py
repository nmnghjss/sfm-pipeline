#!/usr/bin/env python3
"""Correct odom-realtime.csv timestamps using odom-realtime-bk.csv as reference.

The two files describe the same trajectory, so only the constant clock offset
between the two clocks is corrected; the clock rate is assumed to be 1:

    corrected = source + offset

Two ways to pick that offset:

* ``--align first`` (default) ``offset = reference_start - source_start`` -- the
  classic behaviour: only the first sample of each file is used, every other
  sample is shifted by the same amount and all remaining columns are preserved.
* ``--align min-error`` fits the offset on the whole trajectory so that the
  residual trajectory error is minimal, see :func:`compute_offset_min_error`.

The span/rate and the end-point agreement are printed as self-checks, so a
violated scale = 1 assumption (dropped samples, real clock-rate difference) is
visible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# scale = 1 自检阈值（仅告警，不改变校正方式）
RATE_TOL = 5e-4  # 由首尾跨度推算的速率偏差容忍（500 ppm）


def is_number(text: str) -> bool:
    """True when the field can be parsed as a float."""
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def read_table(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    """Read a CSV, returning (header row or None, data rows).

    The first row counts as a header only when its first field is not a number
    (e.g. ``#timestamp, x, y, z, ...``).  A headerless file therefore keeps all
    rows, so the first timestamp gets corrected too instead of being written
    back unchanged.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.reader(file) if row and any(f.strip() for f in row)]

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    header = None
    if not is_number(rows[0][0]):
        header, rows = rows[0], rows[1:]
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return header, rows


def parse_timestamps(rows: list[list[str]], path: Path) -> list[float | int]:
    """Parse the first column and require a strictly increasing sequence.

    Integer timestamps (nanosecond clocks, ~1.8e18) stay ``int``: float64 cannot
    represent them exactly (spacing 256), which would quantise the offset to
    ~256 ns and even swallow sub-256 ns offsets entirely.
    """
    raw = [row[0].strip() for row in rows]
    try:
        values: list[float | int] = [int(value) for value in raw]
    except ValueError:
        try:
            values = [float(value) for value in raw]
        except ValueError as error:
            raise ValueError(f"Invalid timestamp in {path}") from error

    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"Timestamps must be strictly increasing in {path}")
    return values


def compute_offset(source: list[float | int], reference: list[float | int]) -> float | int:
    """Constant time offset that maps the source start onto the reference start."""
    if not source or not reference:
        raise ValueError("Both CSV files must contain at least one data row")
    return reference[0] - source[0]


def shift_timestamps(values: list[float | int], offset: float | int) -> list[float | int]:
    """Apply the constant offset to every timestamp (scale fixed to 1)."""
    return [value + offset for value in values]


def correct_timestamps(source: list[float | int], reference: list[float | int]):
    """Offset from the first samples, scale fixed to 1."""
    return shift_timestamps(source, compute_offset(source, reference))


# ---- 使校正后轨迹误差最小的偏移量 ----------------------------------------
_ALIGN_GRID = 201   # 每轮搜索的候选数
_ALIGN_CHUNK = 64   # 分批评估候选，限制峰值内存
_ALIGN_LEVELS = 4   # 几何收缩轮数（每轮把搜索半径缩到当前步长）


def _median(values: list[float]) -> float:
    """Median of a small list (no numpy needed for this step)."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _parse_xyz(rows: list[list[str]]):
    """(N, 3) x/y/z array from CSV rows, or None when unavailable."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not rows or len(rows[0]) < 4:
        return None
    try:
        return np.array(
            [[float(row[1]), float(row[2]), float(row[3])] for row in rows],
            dtype=float,
        )
    except (ValueError, IndexError):
        return None


def _round_offset(offset: float, source: list[float | int]) -> float | int:
    """Keep the offset integral when the source timestamps are integers."""
    return int(round(offset)) if isinstance(source[0], int) else float(offset)


def _prepare_alignment(source, reference, source_rows, reference_rows, per_second):
    """Time/position arrays for the trajectory-error search, or None.

    Returns ``(t_src, src_rel, t_ref, ref_rel)``; times are seconds relative to
    the reference start (the subtraction happens on ints first, so nanosecond
    stamps keep full precision) and both position arrays share one origin.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    src_xyz = _parse_xyz(source_rows)
    ref_xyz = _parse_xyz(reference_rows)
    if src_xyz is None or ref_xyz is None:
        return None
    if len(src_xyz) != len(source) or len(ref_xyz) != len(reference):
        return None

    def seconds(values: list[float | int]):
        return np.array([value - reference[0] for value in values],
                        dtype=float) / per_second

    return seconds(source), src_xyz - ref_xyz[0], seconds(reference), ref_xyz - ref_xyz[0]


def _alignment_error(offsets, t_src, src_rel, t_ref, ref_rel):
    """Per-candidate ``(mean, max)`` distance from the source to the reference.

    The reference trajectory is sampled at ``t_src + offset``.  Samples whose
    query time falls outside the reference range are dropped instead of being
    clamped, so no phantom error is created at the ends; a candidate that keeps
    no sample at all scores ``inf``.  Candidates are evaluated in chunks so the
    peak memory stays bounded no matter how long the trajectories are.
    """
    import numpy as np

    offsets = np.asarray(offsets, dtype=float)
    means = np.full(len(offsets), np.inf)
    peaks = np.full(len(offsets), np.inf)
    for start in range(0, len(offsets), _ALIGN_CHUNK):
        block = offsets[start:start + _ALIGN_CHUNK][:, None]
        query = t_src[None, :] + block                      # (chunk, N)
        valid = (query >= t_ref[0]) & (query <= t_ref[-1])
        squared = np.zeros_like(query)
        for axis in range(3):
            squared += (np.interp(query, t_ref, ref_rel[:, axis])
                        - src_rel[:, axis]) ** 2
        distance = np.sqrt(squared)
        counts = valid.sum(axis=1)
        totals = np.where(valid, distance, 0.0).sum(axis=1)
        peak = np.where(valid, distance, -np.inf).max(axis=1)
        keep = counts > 0
        stop = start + _ALIGN_CHUNK
        means[start:stop] = np.where(keep, totals / np.maximum(counts, 1), np.inf)
        peaks[start:stop] = np.where(keep, peak, np.inf)
    return means, peaks


def trajectory_error(source, reference, source_rows, reference_rows, offset,
                     per_second):
    """``(mean, max)`` distance from the source samples to the reference path.

    Returns None when x/y/z columns are unavailable or when no sample falls
    inside the reference time range.
    """
    prepared = _prepare_alignment(source, reference, source_rows, reference_rows,
                                  per_second)
    if prepared is None:
        return None
    t_src, src_rel, t_ref, ref_rel = prepared
    means, peaks = _alignment_error([offset / per_second], t_src, src_rel,
                                    t_ref, ref_rel)
    if means[0] == float("inf"):
        return None
    return float(means[0]), float(peaks[0])


def _auto_window_seconds(source: list[float | int], reference: list[float | int],
                         per_second: int) -> float:
    """Refinement search radius = |source duration - reference duration|.

    The seed comes from an index-matched median, so it can only be wrong when
    the two files do not cover the same interval (dropped frames, a late start,
    a truncated tail).  The coverage difference is exactly the size of that
    error, so it is the honest search radius; equal durations mean there is
    nothing to refine (radius 0 -> the seed is kept as is).
    """
    return abs((source[-1] - source[0]) - (reference[-1] - reference[0])) / per_second


def compute_offset_min_error(source, reference, source_rows, reference_rows,
                             per_second, window_seconds: float | None = None):
    """Constant offset that minimises the residual trajectory error.

    Unlike :func:`compute_offset` (first sample only) this uses the whole
    trajectory:

    1. **Seed** from the median of the index-matched residuals
       ``reference[i] - source[i]``.  The median tolerates dropped frames and --
       unlike a position-based search -- does not need the two files to share a
       clock epoch, so it already copes with a source stamped by an unsynced
       device clock months away from the reference.
    2. **Refine** (when x/y/z columns exist) by searching ``seed ±
       window_seconds`` for the offset that minimises the mean Euclidean
       distance between the source positions and the reference trajectory
       interpolated at the corrected timestamps.  ``window_seconds=None`` uses
       :func:`_auto_window_seconds` (the coverage difference), i.e. exactly the
       amount by which an index-matched seed can be off; a ``0`` radius means
       both files cover the same interval and the seed is kept unchanged.

    Integer timestamps get an integer offset back, so the output CSV keeps its
    exact-integer format.  The clock rate is still assumed to be 1.
    """
    if not source or not reference:
        raise ValueError("Both CSV files must contain at least one data row")

    matched = min(len(source), len(reference))
    seed = _median([reference[i] - source[i] for i in range(matched)])

    prepared = _prepare_alignment(source, reference, source_rows, reference_rows,
                                  per_second)
    if prepared is None:
        print("[align] 无可用 x/y/z 列（或缺少 numpy）：只用同索引时间残差中位数，"
              "不做空间精化")
        return _round_offset(seed, source)

    import numpy as np

    t_src, src_rel, t_ref, ref_rel = prepared
    centre = float(seed) / per_second                        # 单位：秒
    radius = (float(window_seconds) if window_seconds is not None
              else _auto_window_seconds(source, reference, per_second))
    source_span = (source[-1] - source[0]) / per_second
    reference_span = (reference[-1] - reference[0]) / per_second
    print(f"[align] seed = {seed} ({seed / per_second:+.6f} s)"
          f"[同索引时间残差中位数]，"
          f"搜索半径 ±{radius:.9f} s = |source 跨度 {source_span:.6f} s - "
          f"reference 跨度 {reference_span:.6f} s|")
    if radius <= 0:
        print("[align] 两文件覆盖时长相同（半径为 0）：无从判断偏移量该往哪边移，"
              "直接采用同索引残差中位数，跳过空间精化")
        return _round_offset(seed, source)
    initial_radius = radius
    hit_boundary = False
    for level in range(_ALIGN_LEVELS):
        grid = np.linspace(centre - radius, centre + radius, _ALIGN_GRID)
        means, _ = _alignment_error(grid, t_src, src_rel, t_ref, ref_rel)
        index = int(np.argmin(means))
        if not np.isfinite(means[index]):
            print(f"[align] 第 {level + 1} 轮搜索区间内所有采样都落在 reference "
                  f"时间范围外，保留上一轮结果")
            break
        if level == 0 and index in (0, _ALIGN_GRID - 1):
            hit_boundary = True
        centre = float(grid[index])
        radius = 2.0 * radius / (_ALIGN_GRID - 1)            # 下一轮只覆盖当前一格

    # 报事实，不下结论：中位数初值本身就是“时间残差”的 L1 最优点，所以只要精化
    # 一动，时间残差必然变大；真正要看的是轨迹误差有没有一起变好（含最大值）。
    fitted = centre * per_second
    probes = np.array([seed / per_second, fitted / per_second])
    probe_means, probe_peaks = _alignment_error(probes, t_src, src_rel, t_ref, ref_rel)
    print(f"[align] 空间精化移动偏移量 {(fitted - seed) / per_second:+.6f} s；"
          f"平均轨迹误差 {probe_means[0]:.4f} -> {probe_means[1]:.4f} m，"
          f"最大 {probe_peaks[0]:.4f} -> {probe_peaks[1]:.4f} m")
    if hit_boundary:
        print(f"[warn] 拟合值落在搜索窗口边界（±{initial_radius:.6f} s="
              f"|source 跨度 - reference 跨度|）：真实最优可能在窗口外，"
              f"可用 --align-window 放宽后重试")
    return _round_offset(fitted, source)


def format_timestamp(value: float | int) -> str:
    """Integer timestamps verbatim; fractional (seconds) inputs keep 9 decimals."""
    return f"{value:d}" if isinstance(value, int) else f"{value:.9f}"


def write_csv(path: Path, header: list[str] | None, rows: list[list[str]],
              values: list[float | int]) -> None:
    """Write corrected timestamps, preserving the header and all other columns."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        if header is not None:
            writer.writerow(header)
        for row, value in zip(rows, values):
            row = list(row)
            row[0] = format_timestamp(value)
            writer.writerow(row)


def visualize(
    source: list[float | int],
    reference: list[float | int],
    corrected: list[float | int],
    source_rows: list[list[str]],
    reference_rows: list[list[str]],
    per_second: int,
    save_path: Path | None,
    show: bool,
) -> None:
    """Plot the source/reference trajectory and the residual after correction.

    Four panels:
      1. X-Y top view of the reference and source paths.
      2. X/Y/Z over time -- raw source (pre-offset, dotted), corrected source
         (dashed) and reference (solid), all in the reference clock.
      3. Timestamp residual ``source - reference`` over time, before vs after.
      4. Position residual in metres, obtained by interpolating the reference
         trajectory at the (raw or corrected) source timestamps.

    Labels are English: the default Matplotlib fonts have no CJK glyphs, so
    Chinese text would be drawn as empty boxes.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        print(f"[warn] 可视化需要 matplotlib 与 numpy，已跳过：{error}")
        return

    base = reference[0]

    def to_seconds(values: list[float | int]) -> "np.ndarray":
        # Subtract the reference start *before* casting to float: the raw
        # nanosecond stamps (~1.8e18) are not exactly representable in float64.
        return np.array([value - base for value in values], dtype=float) / per_second

    def xyz(rows: list[list[str]]):
        """(N, 3) x/y/z array, or None when the CSV has no coordinate columns."""
        return _parse_xyz(rows)

    ref_t = to_seconds(reference)
    raw_t = to_seconds(source)
    cor_t = to_seconds(corrected)

    matched = min(len(ref_t), len(cor_t))
    residual_before = raw_t[:matched] - ref_t[:matched]
    residual_after = cor_t[:matched] - ref_t[:matched]

    # "Before correction" only means something when both clocks share an epoch.
    # If either side carries a device clock (unsynced RTC / uptime counter), the
    # offset can be months wide, and plotting it would stretch every time axis
    # until the real drift became invisible.
    raw_comparable = bool(raw_t[0] <= ref_t[-1] and raw_t[-1] >= ref_t[0])

    def annotate_uncomparable(ax) -> None:
        ax.text(0.02, 0.92, "raw source not comparable\n(different clock epoch)",
                transform=ax.transAxes, fontsize=8, va="top", color="tab:red")

    ref_xyz = xyz(reference_rows)
    src_xyz = xyz(source_rows)
    has_xyz = (
        ref_xyz is not None
        and src_xyz is not None
        and len(src_xyz) == len(raw_t)
        and len(ref_xyz) == len(ref_t)
    )
    pos_before = pos_after = None
    pos_t_before = pos_t_after = None
    if has_xyz:
        ref_rel = ref_xyz - ref_xyz[0]
        src_rel = src_xyz - ref_xyz[0]

        def sample_at(query: "np.ndarray") -> "np.ndarray":
            """Reference position interpolated at the query times.

            Queries outside the reference time range become NaN instead of the
            clamped edge value: the source file may be longer than the
            reference, and clamping would fabricate a large constant "error".
            """
            values = np.stack(
                [np.interp(query, ref_t, ref_rel[:, axis]) for axis in range(3)],
                axis=1,
            )
            values[(query < ref_t[0]) | (query > ref_t[-1])] = np.nan
            return values

        pos_before = np.linalg.norm(src_rel - sample_at(raw_t), axis=1)
        pos_after = np.linalg.norm(src_rel - sample_at(cor_t), axis=1)
        # The residual arrays are indexed like the source, so plot them on the
        # source time axes -- the reference axis may have a different length.
        pos_t_before, pos_t_after = raw_t, cor_t

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Odom timestamp correction: source vs reference", fontweight="bold")

    # 1) Top view of the two trajectories.
    ax = axes[0, 0]
    if has_xyz:
        ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], "-", color="tab:blue", label="reference")
        ax.plot(src_xyz[:, 0], src_xyz[:, 1], "--", color="tab:orange", label="source")
        ax.plot(ref_xyz[0, 0], ref_xyz[0, 1], "o", color="tab:blue", ms=6,
                label="reference start")
        ax.plot(src_xyz[0, 0], src_xyz[0, 1], "x", color="tab:orange", ms=7,
                label="source start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    else:
        ax.text(0.5, 0.5, "no x/y/z columns in CSV", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Trajectory (top view)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 2) Position over time: shows the time shift applied to the source.
    ax = axes[0, 1]
    if has_xyz:
        axis_names = ("x", "y", "z")
        for axis in range(3):
            color = f"C{axis}"
            offset_axis = ref_xyz[0, axis]
            ax.plot(ref_t, ref_xyz[:, axis] - offset_axis, "-", color=color,
                    label=f"reference {axis_names[axis]}")
            ax.plot(cor_t, src_xyz[:, axis] - offset_axis, "--", color=color,
                    alpha=0.85, label=f"corrected {axis_names[axis]}")
            if raw_comparable:
                ax.plot(raw_t, src_xyz[:, axis] - offset_axis, ":", color=color,
                        alpha=0.6)
        ax.legend(loc="best", fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "no x/y/z columns in CSV", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Position over time"
                 + (" (dotted = raw source)" if raw_comparable
                    else " (raw source omitted)"))
    ax.set_xlabel("time since reference start (s)")
    ax.set_ylabel("position (m)")
    ax.grid(True, alpha=0.3)

    # 3) Residual of the alignment in the time domain.
    ax = axes[1, 0]
    if raw_comparable:
        ax.plot(ref_t[:matched], residual_before, color="tab:red",
                label="before correction")
    else:
        annotate_uncomparable(ax)
    ax.plot(ref_t[:matched], residual_after, color="tab:green",
            label="after correction")
    # 零参考线只做参考，不让它参与自动缩放：axhline 会把自己的 y=0 算进数据范围，
    # 把 y 轴钉成从 0 开始。残差整体偏离 0 时（例如 --align min-error 校正了真实的
    # 位姿时延，残差恒为 +1.15 s）曲线就会被压成一条贴着上边的直线，看不出结构。
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    drawn = ([residual_before, residual_after] if raw_comparable
             else [residual_after])
    drawn_values = np.concatenate(drawn)
    low, high = float(drawn_values.min()), float(drawn_values.max())
    pad = 0.05 * (high - low) if high > low else max(abs(high) * 0.05, 1e-9)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title("Timestamp residual: source - reference")
    ax.set_xlabel("reference time (s)")
    ax.set_ylabel("residual (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # 4) Residual of the alignment in the spatial domain.
    ax = axes[1, 1]
    if pos_before is not None:
        if np.any(np.isfinite(pos_before)):
            ax.plot(pos_t_before, pos_before, color="tab:red",
                    label="before correction")
        else:
            annotate_uncomparable(ax)
        ax.plot(pos_t_after, pos_after, color="tab:green", label="after correction")
        ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no x/y/z columns in CSV", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Position residual vs reference path")
    ax.set_xlabel("time since reference start (s)")
    ax.set_ylabel("error (m)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"[vis] figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[vis] timestamp residual after correction: "
          f"max {np.max(np.abs(residual_after)):.6e} s, "
          f"mean {np.mean(np.abs(residual_after)):.6e} s")
    if pos_before is not None:
        def finite_max(values: "np.ndarray") -> float:
            inside = values[np.isfinite(values)]
            return float(inside.max()) if inside.size else float("nan")

        before_max = finite_max(pos_before)
        after_max = finite_max(pos_after)
        if np.isfinite(before_max):
            print(f"[vis] position residual: before max {before_max:.4f} m, "
                  f"after max {after_max:.4f} m")
        else:
            print(f"[vis] position residual after correction: max {after_max:.4f} m")
        # 位置残差同时包含“时间没对齐”和“两条轨迹本身就不同”两部分；同索引直接
        # 相减（不做任何插值/时间校正）能把后者单独量出来。
        direct = float(np.linalg.norm(src_rel[:matched] - ref_rel[:matched], axis=1).max())
        print(f"[vis] 同索引直接相减（不插值）位置差最大 {direct:.4f} m："
              f"这部分来自两条轨迹自身的估计差异，与时间戳校正无关")
        outside = int(np.count_nonzero(~np.isfinite(pos_before)))
        if not raw_comparable:
            print(f"[vis] raw source 与 reference 不同时钟纪元（首行相差约 "
                  f"{abs(raw_t[0]):.3g} s），“前”曲线不可比，已在图中省略")
        elif outside:
            print(f"[vis] {outside}/{len(pos_before)} 个 source 采样落在 reference 时间范围外，"
                  f"位置残差图中已断开（插值不外推，避免端点假误差）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("odom-realtime.csv"),
        help="CSV whose timestamps should be corrected (default: odom-realtime.csv)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("odom.csv"),
        help="Reference CSV (default: odom.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("odom-realtime-corrected.csv"),
        help="Output CSV (default: odom-realtime-corrected.csv)",
    )
    parser.add_argument(
        "--align",
        choices=("first", "min-error"),
        default="first",
        help="偏移量取法：first=只用两文件首行（默认，与旧行为一致）；"
             "min-error=用整条轨迹拟合，使校正后的轨迹误差最小",
    )
    parser.add_argument(
        "--align-window",
        type=float,
        default=None,
        metavar="SECONDS",
        help="min-error 时的空间搜索半径（秒）；默认 auto = |source 持续时长 - "
             "reference 持续时长|（= 按索引对应可能错多少的上界）",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="显示 source/reference 轨迹与校正误差（需要 matplotlib）",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="把可视化结果保存为图片（例如 vis.png），可与 --vis 同时使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_header, reference_rows = read_table(args.reference)
    source_header, source_rows = read_table(args.source)
    reference = parse_timestamps(reference_rows, args.reference)
    source = parse_timestamps(source_rows, args.source)

    # 纳秒时间戳（≈1e18）还是秒时间戳（≈1e9），用于换算、搜索与可视化
    per_second = 10 ** 9 if abs(source[0]) > 10 ** 13 else 1

    if args.align == "min-error":
        offset = compute_offset_min_error(source, reference, source_rows,
                                          reference_rows, per_second,
                                          args.align_window)
    else:
        offset = compute_offset(source, reference)
    corrected = shift_timestamps(source, offset)
    write_csv(args.output, source_header, source_rows, corrected)

    source_span = source[-1] - source[0]
    reference_span = reference[-1] - reference[0]
    drift = corrected[-1] - reference[-1]

    # 同索引残差对行数不敏感：任一文件多出/少掉若干采样都不影响它，因此它是
    # 判断“常量偏移是否够用”的首选指标。跨度比只在行数相同时才有速率含义。
    matched = min(len(source), len(reference))
    index_residual = [corrected[i] - reference[i] for i in range(matched)]
    max_residual = max(abs(value) for value in index_residual)

    print(f"Reference: {args.reference} ({len(reference)} rows)")
    print(f"Source:    {args.source} ({len(source)} rows)")
    print(f"Header:    reference={'yes' if reference_header else 'no'}, "
          f"source={'yes' if source_header else 'no'}")
    print(f"Output:    {args.output}")
    print(f"Alignment: {args.align}")
    print(f"Offset (constant): {offset}  ({offset / per_second:.6f} s)")
    print(f"Mapping: corrected = source + {offset}  (scale = 1)")
    print(f"Corrected range: {corrected[0]} -> {corrected[-1]}")

    # ---- min-error：与首行取法对比，并给出两种取法的轨迹误差 ------------
    if args.align == "min-error":
        baseline = compute_offset(source, reference)
        baseline_corrected = shift_timestamps(source, baseline)
        baseline_residual = max(abs(baseline_corrected[i] - reference[i])
                                for i in range(matched))
        print(f"Align ref:   首行取法 offset = {baseline} "
              f"({baseline / per_second:.6f} s)，与拟合值相差 "
              f"{(offset - baseline) / per_second:.6f} s")
        print(f"Align index: 同索引最大残差 首行 {baseline_residual / per_second:.6f} s"
              f" -> 拟合 {max_residual / per_second:.6f} s")
        for label, candidate in (("first", baseline), ("fitted", offset)):
            error = trajectory_error(source, reference, source_rows, reference_rows,
                                     candidate, per_second)
            if error is not None:
                mean_error, max_error = error
                print(f"Align err [{label}]: 平均轨迹误差 {mean_error:.4f} m, "
                      f"最大 {max_error:.4f} m")
        if args.align_window is None:
            window_text = (f"auto = |{source_span / per_second:.6f} - "
                           f"{reference_span / per_second:.6f}| = "
                           f"{abs(source_span - reference_span) / per_second:.6f} s")
        else:
            window_text = f"{args.align_window:g} s"
        print(f"Align search: 空间搜索半径 {window_text}，目标 = 平均轨迹误差最小")

    # ---- scale = 1 自检（只告警，不改变校正方式） ------------------------
    if len(source) != len(reference):
        print(f"[warn] 行数不一致（source {len(source)} vs reference {len(reference)}）："
              f"行数差异通常意味着丢帧或采样率不同；"
              f"此时首尾跨度/末点比对会被“覆盖时长不同”污染，改按同索引比较")
    print(f"Check [index]: |corrected - reference| 同索引最大残差 {max_residual} "
          f"({max_residual / per_second:.6f} s)")

    if source_span:
        rate = reference_span / source_span
        print(f"Check [span]: {source_span} vs {reference_span} -> implied rate {rate:.9f}")
        print(f"Check [end]:  corrected_end - reference_end = {drift} "
              f"({drift / per_second:.6f} s, {100 * abs(drift / source_span):.4f}% of span)")
        if abs(rate - 1) > RATE_TOL:
            if len(source) != len(reference):
                print(f"[info] 跨度比 {rate:.9f} 主要反映两文件覆盖时长不同（行数不等），"
                      f"不能当作时钟速率差；真实对齐误差见 Check [index]")
            else:
                print(f"[warn] 速率偏差 {abs(rate - 1):.2e} 超过阈值 {RATE_TOL:.0e}："
                      f"两端时钟速率可能不同（本脚本按 scale = 1 校正，未补偿速率差）")
    else:
        print("[warn] source 时间跨度为 0，跳过 span/rate 自检")

    # ---- 可视化（source/reference 轨迹 + 校正误差） --------------------
    if args.vis or args.save is not None:
        visualize(source, reference, corrected, source_rows, reference_rows,
                  per_second, args.save, args.vis)


if __name__ == "__main__":
    main()