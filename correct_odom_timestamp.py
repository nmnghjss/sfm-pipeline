#!/usr/bin/env python3
"""Correct odom-realtime.csv timestamps using odom-realtime-bk.csv as reference.

The two files are expected to describe the same trajectory in sampling order.
The correction is an affine time mapping based on the first and last samples:

    corrected = reference_start + (source - source_start) * scale

This handles both a constant clock offset and a small clock-rate difference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a CSV while retaining the original header/comment and fields."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    header = rows[0]
    data = [row for row in rows[1:] if row and any(field.strip() for field in row)]
    if not data:
        raise ValueError(f"CSV has no data rows: {path}")
    if any(len(row) == 0 for row in data):
        raise ValueError(f"CSV contains an invalid empty row: {path}")
    return header, data


def timestamp_values(rows: list[list[str]], path: Path) -> list[float]:
    """Parse the first column as float and require a monotonic time sequence."""
    try:
        values = [float(row[0].strip()) for row in rows]
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid timestamp in {path}") from error

    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"Timestamps must be strictly increasing in {path}")
    return values


def correct_timestamps(source: list[float], reference: list[float]) -> list[float]:
    """Map source timestamps onto the reference start/end time axis."""
    if len(source) < 2 or len(reference) < 2:
        raise ValueError("Both CSV files must contain at least two data rows")

    source_start, source_end = source[0], source[-1]
    reference_start, reference_end = reference[0], reference[-1]
    scale = (reference_end - reference_start) / (source_end - source_start)
    return [reference_start + (stamp - source_start) * scale for stamp in source]


def write_csv(path: Path, header: list[str], rows: list[list[str]], values: list[float]) -> None:
    """Write corrected timestamps and preserve all remaining columns."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        for row, value in zip(rows, values):
            row = list(row)
            row[0] = f"{value:.0f}"
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
    reference_header, reference_rows = read_csv_rows(args.reference)
    source_header, source_rows = read_csv_rows(args.source)
    reference = timestamp_values(reference_rows, args.reference)
    source = timestamp_values(source_rows, args.source)
    corrected = correct_timestamps(source, reference)
    write_csv(args.output, source_header, source_rows, corrected)

    scale = (reference[-1] - reference[0]) / (source[-1] - source[0])
    offset = reference[0] - source[0] * scale
    print(f"Reference: {args.reference} ({len(reference)} rows)")
    print(f"Source:    {args.source} ({len(source)} rows)")
    print(f"Output:    {args.output}")
    print(f"Mapping: corrected = {scale:.12f} * source + {offset:.3f}")
    print(f"Corrected range: {corrected[0]:.0f} -> {corrected[-1]:.0f}")


if __name__ == "__main__":
    main()