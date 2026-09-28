"""Turn benchmark JSONL into markdown tables. The only place numbers are computed from raw data.

    python benchmarks/laya/report.py benchmarks/laya/results/*.jsonl

Percentiles are nearest-rank. The run-to-run gate compares p50 across measured runs (every run whose
label is not "feasibility") of the same config, workload and concurrency: (max - min) / min <= 10%.
In-process results have concurrency 1.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict

GATE = 0.10
PHASES = ["import_s", "load_s", "process_to_ready_s", "warmup_s"]  # whichever a result file has


def percentile(sorted_values, p):
    return sorted_values[max(0, math.ceil(p * len(sorted_values)) - 1)]


def read(paths):
    records = []
    for path in paths:
        with open(path) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def short(sha, dirty=False):
    return (sha or "?")[:7] + ("+dirty" if dirty else "")


def environment(envs):
    rows = []
    for k, e in sorted(envs.items()):
        dtype = f"{e['weights_dtype']} (amp {e['amp_dtype']}, >= {e['mps_amp_min_rows']} rows)"
        if e.get("dtype_source"):
            dtype += f" [{e['dtype_source']}]"
        rows.append(
            [
                *k,
                e["device_actual"],
                dtype,
                e["chip"],
                e["os"],
                e["power"],
                e["loadavg_1m"],
                f"laya {e['laya']} / torch {e['torch']}",
                short(e["checkpoint_revision"]),
                short(e["omni_sha"], e["omni_dirty"]),
            ]
        )
    headers = [
        "config",
        "run",
        "device",
        "weights (autocast)",
        "chip",
        "os",
        "power",
        "load 1m",
        "versions",
        "ckpt",
        "omni",
    ]
    return table(headers, rows)


def phase_table(phases):
    present = [p for p in PHASES if any(ph.get(p) is not None for ph in phases.values())]
    workload_ids = sorted({w for ph in phases.values() for w in ph["first_ms"]})
    rows = [
        [*k, *(ph.get(p) for p in present), *(ph["first_ms"].get(w) for w in workload_ids)]
        for k, ph in sorted(phases.items())
    ]
    headers = ["config", "run", *(p.removesuffix("_s") for p in present), *(f"first {w}" for w in workload_ids)]
    return table(headers, rows)


def latency(records):
    samples = defaultdict(list)
    for r in records:
        if r["type"] == "req" and r.get("status", 200) == 200:
            samples[(r["config"], r["workload"], r.get("concurrency", 1), r["run"])].append(r["wall_ms"])
    rows, p50s = [], defaultdict(dict)
    for (config, workload, concurrency, run), values in sorted(samples.items()):
        values.sort()
        mean = statistics.fmean(values)
        cv = statistics.stdev(values) / mean if len(values) > 1 else 0.0
        p50 = percentile(values, 0.50)
        if run != "feasibility":
            p50s[(config, workload, concurrency)][run] = p50
        rows.append(
            [
                config,
                workload,
                concurrency,
                run,
                len(values),
                round(p50, 2),
                round(percentile(values, 0.95), 2),
                round(mean, 2),
                f"{cv:.1%}",
            ]
        )
    return table(["config", "workload", "conc", "run", "n", "p50", "p95", "mean", "CV"], rows), p50s


def gate(p50s):
    rows = []
    for (config, workload, concurrency), by_run in sorted(p50s.items()):
        if len(by_run) < 2:
            rows.append([config, workload, concurrency, len(by_run), "", "needs 2 measured runs"])
            continue
        spread = (max(by_run.values()) - min(by_run.values())) / min(by_run.values())
        rows.append([config, workload, concurrency, len(by_run), f"{spread:.1%}", "PASS" if spread <= GATE else "FAIL"])
    return table(["config", "workload", "conc", "runs", "spread", "gate"], rows)


def throughput(records):
    rows = [
        [r["config"], r["workload"], r["concurrency"], r["run"], r["n"], r["errors"], r["elapsed_s"], r["rps"]]
        for r in records
        if r["type"] == "throughput"
    ]
    return table(["config", "workload", "conc", "run", "n", "errors", "elapsed s", "req/s"], sorted(rows))


def memory(phases, ends):
    """Physical footprint (Activity Monitor's "Memory", includes Metal allocations) of the process running
    Laya: the benchmark itself in-process, the worker over HTTP. Peak is the process lifetime maximum."""
    rows = []
    for k in sorted(ends):
        ph, e = phases.get(k, {}), ends[k]
        rows.append(
            [
                *k,
                ph.get("footprint_mb"),
                e.get("footprint_mb"),
                e.get("footprint_peak_mb"),
                ph.get("mps_driver_mb"),
                e.get("mps_driver_mb"),
                e.get("frontend_footprint_mb"),
            ]
        )
    headers = [
        "config",
        "run",
        "footprint after warmup",
        "footprint end",
        "footprint peak",
        "MPS driver after warmup",
        "MPS driver end",
        "frontend footprint",
    ]
    return table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    records = read(args.files)

    def by_run(kind):
        return {(r["config"], r["run"]): r for r in records if r["type"] == kind}

    envs, phases, ends = by_run("env"), by_run("phase"), by_run("end")
    latency_table, p50s = latency(records)
    print("## Environment\n\n" + environment(envs))
    print("\n## Phases (s) and first request per workload (ms)\n\n" + phase_table(phases))
    print("\n## Warm latency (ms)\n\n" + latency_table)
    print(f"\n## Run-to-run gate (p50 spread across measured runs <= {GATE:.0%})\n\n" + gate(p50s))
    if any(r["type"] == "throughput" for r in records):
        print("\n## Throughput\n\n" + throughput(records))
    print("\n## Memory (MB)\n\n" + memory(phases, ends))


if __name__ == "__main__":
    main()
