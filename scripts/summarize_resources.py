#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path


UNIT_TO_MIB = {
    "B": 1 / (1024 * 1024),
    "KiB": 1 / 1024,
    "MiB": 1,
    "GiB": 1024,
    "TiB": 1024 * 1024,
}


def parse_float(value):
    if value is None:
        return None
    value = value.strip().replace("%", "")
    if not value or value in {"[N/A]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_memory_mib(value):
    if value is None:
        return None
    match = re.match(r"\s*([0-9.]+)\s*([KMGT]?i?B)\s*", value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    multiplier = UNIT_TO_MIB.get(unit)
    if multiplier is None:
        return None
    return amount * multiplier


def max_value(rows, field):
    values = [parse_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def first_nonempty(rows, field):
    for row in rows:
        value = row.get(field, "").strip()
        if value:
            return value
    return None


def summarize(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    gpu_memory_available = any(
        row.get("gpu_memory_available", "").strip().lower() == "true" for row in rows
    )
    gpu_memory_note = None
    if not gpu_memory_available:
        gpu_memory_note = first_nonempty(rows, "gpu_memory_note")

    docker_mem_values = [
        parse_memory_mib(row.get("docker_mem_usage", "").split("/", 1)[0]) for row in rows
    ]
    docker_mem_values = [value for value in docker_mem_values if value is not None]

    summary = {
        "source_csv": str(csv_path),
        "sample_count": len(rows),
        "started_at": rows[0]["timestamp"] if rows else None,
        "finished_at": rows[-1]["timestamp"] if rows else None,
        "gpu": {
            "name": first_nonempty(rows, "gpu_name"),
            "peak_util_percent": max_value(rows, "gpu_util_percent"),
            "peak_power_w": max_value(rows, "gpu_power_w"),
            "max_temperature_c": max_value(rows, "gpu_temp_c"),
            "memory_accounting_available": gpu_memory_available,
            "memory_note": gpu_memory_note,
            "peak_memory_util_percent": max_value(rows, "gpu_mem_util_percent")
            if gpu_memory_available
            else None,
            "peak_memory_used_mib": max_value(rows, "gpu_mem_used_mib")
            if gpu_memory_available
            else None,
        },
        "host": {
            "peak_load_1m": max_value(rows, "load_1m"),
            "peak_memory_used_mib": max_value(rows, "mem_used_mib"),
            "memory_total_mib": max_value(rows, "mem_total_mib"),
        },
        "container": {
            "peak_cpu_percent": max_value(rows, "docker_cpu_percent"),
            "peak_memory_used_mib": max(docker_mem_values) if docker_mem_values else None,
            "max_pids": max_value(rows, "docker_pids"),
        },
    }

    return summary


def main():
    if len(sys.argv) != 3:
        print(
            "usage: summarize_resources.py RESOURCE_CSV SUMMARY_JSON",
            file=sys.stderr,
        )
        return 2

    csv_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])
    summary = summarize(csv_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Resource summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
