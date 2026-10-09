#!/usr/bin/env python3
"""Correct odom-realtime.csv timestamps using odom-realtime-bk.csv as reference.

The two files describe the same trajectory, so only the constant clock offset
between the two clocks is corrected; the clock rate is assumed to be 1:

    corrected = source + (reference_start - source_start)

Only the first sample of each file is used; every other sample is shifted by the
same amount and all remaining columns are preserved. The span/rate and the
end-point agreement are printed as self-checks, so a violated scale = 1
assumption (dropped samples, real clock-rate difference) is visible.
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
        default=Path("odom-realtime-bk.csv"),
        help="Reference CSV (default: odom-realtime-bk.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("odom-realtime-corrected.csv"),
        help="Output CSV (default: odom-realtime-corrected.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_header, reference_rows = read_table(args.reference)
    source_header, source_rows = read_table(args.source)
    reference = parse_timestamps(reference_rows, args.reference)
    source = parse_timestamps(source_rows, args.source)

    offset = compute_offset(source, reference)
    corrected = shift_timestamps(source, offset)
    write_csv(args.output, source_header, source_rows, corrected)

    # 纳秒时间戳（≈1e18）还是秒时间戳（≈1e9），仅用于打印换算
    per_second = 10 ** 9 if abs(source[0]) > 10 ** 13 else 1
    source_span = source[-1] - source[0]
    reference_span = reference[-1] - reference[0]
    drift = corrected[-1] - reference[-1]

    print(f"Reference: {args.reference} ({len(reference)} rows)")
    print(f"Source:    {args.source} ({len(source)} rows)")
    print(f"Header:    reference={'yes' if reference_header else 'no'}, "
          f"source={'yes' if source_header else 'no'}")
    print(f"Output:    {args.output}")
    print(f"Offset (constant): {offset}  ({offset / per_second:.6f} s)")
    print(f"Mapping: corrected = source + {offset}  (scale = 1)")
    print(f"Corrected range: {corrected[0]} -> {corrected[-1]}")

    # ---- scale = 1 自检（只告警，不改变校正方式） ------------------------
    if len(source) != len(reference):
        print(f"[warn] 行数不一致（source {len(source)} vs reference {len(reference)}）："
              f"常量偏移取自首行，行数差异通常意味着丢帧或采样率不同")
    if source_span:
        rate = reference_span / source_span
        print(f"Check [span]: {source_span} vs {reference_span} -> implied rate {rate:.9f}")
        print(f"Check [end]:  corrected_end - reference_end = {drift} "
              f"({drift / per_second:.6f} s, {100 * abs(drift / source_span):.4f}% of span)")
        if abs(rate - 1) > RATE_TOL:
            print(f"[warn] 速率偏差 {abs(rate - 1):.2e} 超过阈值 {RATE_TOL:.0e}："
                  f"两端时钟速率可能不同（本脚本按 scale = 1 校正，未补偿速率差）")
    else:
        print("[warn] source 时间跨度为 0，跳过 span/rate 自检")


if __name__ == "__main__":
    main()