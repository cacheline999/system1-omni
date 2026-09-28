# Laya on Apple Silicon

This recipe serves Laya on the GPU of an Apple Silicon Mac (PyTorch MPS) with the Laya worker
from [`src/models/laya/`](../../src/models/laya/), puts the Rust frontend in front of it and runs
the benchmark suite. The [Laya text worker](README.md) recipe covers the plain CPU setup.

Validated on an M1 Pro (16 GB, 16-core GPU), macOS 26.1, Python 3.12, `laya[serve]==0.3.20`,
torch 2.14.0 and the `english` checkpoint (`convaiinnovations/laya` at `55cf4c4`). Other M-series
Macs have not been tested.

Run all commands from the repository root.

## Install

Use Python 3.12. If `python3.12` is not on your `PATH`, install it with `brew install python@3.12`
or `uv python install 3.12`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'laya[serve]==0.3.20' pytest
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

The last command must print `True`. The standard macOS arm64 wheel of torch includes MPS.

## Start the worker

```sh
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=mps LAYA_MODELS=english \
LAYA_REQUIRE_DEVICE=1 \
  .venv/bin/python src/models/laya/worker.py
```

First startup downloads the checkpoint (about 850 MB). The worker loads the model, runs a warmup
over short, long and multi-question requests, and only then listens on port 8000, so the first
request it accepts is already warm. `LAYA_REQUIRE_DEVICE=1` makes it exit instead of silently
serving on the CPU when the model cannot be placed on MPS.

Check what it is running on:

```sh
curl -s http://127.0.0.1:8000/health
```

`device` must be `mps` and `device_mismatch` `false`. The response also names the checkpoint and
revision, the weight dtype (`torch.float32`; Laya upcasts the fp16 checkpoint on MPS), the autocast
dtype Laya uses for requests with at least `mps_amp_min_rows` questions, and the warmup time.

### Compiled one-question path

```sh
LAYA_WORKER_COMPILE=single LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=mps \
LAYA_MODELS=english LAYA_REQUIRE_DEVICE=1 \
  .venv/bin/python src/models/laya/worker.py
```

`single` sends requests with one question through a `torch.compile` graph and runs the rest eagerly.
In two measured runs on the M1 Pro it cut warm p50 for a 68-token one-question request from about
46 ms to 33 ms (−29%) and for a 47-token one from about 39 ms to 26 ms, left three- and six-question
requests within about 3%, and gave the same answers as CPU within the benchmark tolerances. The price is
startup: the worker was ready after about 22 s instead of 8 s while the warmup compiles, and the
gain shrinks with length (−7% to −13% at 484 tokens).

`/health` reports under `compile` how many graphs existed when the worker became ready and how many
exist now; `recompiled_after_ready: true` means a request shape was not covered by the warmup. `all`
compiles every path; in feasibility runs it made multi-question requests up to 65% slower and took
over a minute to start, so it is not recommended.

## Start the frontend

In another terminal:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

## Send a request

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

The frontend forwards the worker's response unchanged; `compare_with_backend.py` from the
[Laya text worker](README.md#compare-responses) recipe checks that against this setup as well.

## Test

```sh
.venv/bin/python -m pytest src/models/laya/tests                    # unit tests, no model
LAYA_CONTRACT=1 .venv/bin/python -m pytest src/models/laya/tests    # plus contract tests against a CPU worker
```

## Benchmark

Stop the worker and frontend first; the benchmark starts its own. The suite and its measurement
rules are described in [`benchmarks/laya/`](../../benchmarks/laya/README.md). A first pass that
checks everything runs:

```sh
.venv/bin/python benchmarks/laya/check_workloads.py
.venv/bin/python benchmarks/laya/bench_inproc.py --device mps --config C2 --run feasibility
.venv/bin/python benchmarks/laya/bench_http.py --config C3 --run feasibility --spawn .venv/bin/laya-serve
.venv/bin/python benchmarks/laya/bench_http.py --config C4 --run feasibility \
  --url http://127.0.0.1:8080 --frontend target/release/omni-jev --spawn .venv/bin/laya-serve
.venv/bin/python benchmarks/laya/report.py benchmarks/laya/results/*.jsonl
```

Runs labelled anything other than `feasibility` refuse to start on battery power or when the
1-minute load average is above 2, so close other heavy applications and plug the Mac in first.

## Troubleshooting

- `device_mismatch: true`, or the worker exits with `asked for mps, model is on cpu`: MPS is not
  available to this Python. Check the `torch.backends.mps.is_available()` line above; an x86_64
  Python running under Rosetta cannot use MPS.
- The worker process uses about 4 GB (Activity Monitor's Memory column, which counts MPS
  allocations). On a 16 GB Mac, close other large applications before benchmarking.
- `Address already in use`: another worker or frontend still holds port 8000 or 8080.
