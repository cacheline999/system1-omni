# Laya benchmarks

Measures Laya on CPU and Apple Silicon (MPS) for [#3](https://github.com/ThinkFlowLab/system1-omni/issues/3).
Every script writes raw JSONL; `report.py` is the only place numbers are computed.

| file | purpose |
| --- | --- |
| `workloads.src.py` → `workloads.jsonl` | fixed inputs: W1–W6 timed, P* parity-only |
| `check_workloads.py` | tokens per row with Laya's tokenizer, ±10% of each target |
| `bench_inproc.py` | in-process: import, load, warmup, first request per workload, warm latency, memory |
| `bench_http.py` | against a `/v1/systemone` worker: process-to-ready (with `--spawn`), first request per workload, warm latency and throughput at each `--concurrency` |
| `profile_mps.py` | where a request's time goes on MPS: length sweep with a fixed-cost fit, stage split (encode, dispatch, GPU wait, copy back, decode) and host operator counts |
| `parity.py` | answers of every run vs a reference run, tolerances fixed in advance (fp32 1e-3, fp16 1e-2) |
| `env.py` | run header: SHAs, versions, checkpoint revision, hardware, power, load |
| `report.py` | JSONL → tables, including the run-to-run gate |

## Run

Use the same environment as the Laya recipe (`laya[serve]==0.3.20`, Python 3.12).

```sh
python benchmarks/laya/check_workloads.py
python benchmarks/laya/bench_inproc.py --device cpu --config C1 --run feasibility
python benchmarks/laya/bench_inproc.py --device mps --config C2 --run feasibility
python benchmarks/laya/bench_inproc.py --device mps --config C2 --run m1
python benchmarks/laya/bench_inproc.py --device mps --config C2 --run m2
python benchmarks/laya/bench_http.py --config C3 --run feasibility --spawn .venv/bin/laya-serve
python benchmarks/laya/bench_http.py --config C4 --run feasibility --url http://127.0.0.1:8080
python benchmarks/laya/report.py benchmarks/laya/results/*.jsonl
python benchmarks/laya/parity.py benchmarks/laya/results/*.jsonl --ref C1
```

Measured runs (any `--run` other than `feasibility`) refuse to start on battery power or when the
1-minute load average is above `--max-load` (default 2). Two measured runs of a config pass when
their p50s differ by at most 10% for every workload.

Timing: wall clock around `Agent.system_one` followed by `torch.mps.synchronize()`, so it includes
tokenization and post-processing. Workload order is shuffled per run with `--seed`.

Memory is the physical footprint of the process running Laya (`proc_pid_rusage`, the same number as
Activity Monitor's "Memory" and `footprint -p`). On Apple silicon it includes Metal allocations, so
in-process and worker numbers are comparable and MPS tensors are counted.
