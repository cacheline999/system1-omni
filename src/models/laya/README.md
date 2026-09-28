# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

Status: the Python worker below serves LAYA through laya-serve on CPU and Apple Silicon (PyTorch MPS,
validated on an M1 Pro). No native CUDA or Metal backend yet.

## Worker

`worker.py` runs laya-serve (`laya[serve]==0.3.20`) with two changes:

- It binds only after a warmup over short, long and multi-question requests, so `/health` never
  answers for a worker that has not run a forward pass. Without it the first request after `/health`
  returned 200 took ~230 ms against ~45 ms warm on an M1 Pro (MPS); with it, ~73 ms.
- `/health` reports the device, weight and autocast dtypes of the loaded model, the checkpoint and
  revision, and `device_mismatch` when the model is not on the device `LAYA_DEVICE` asked for.
  laya-serve reports `LAYA_DEVICE` as configured, and laya falls back to CPU with only a printed
  warning. `LAYA_REQUIRE_DEVICE=1` makes the worker exit instead.

Configuration is laya-serve's (`LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_MODELS`, `LAYA_API_KEY`, ...),
plus `LAYA_WORKER_COMPILE=off|single|all`: `single` sends one-question requests through a
`torch.compile(dynamic=True)` graph compiled during warmup and runs the rest eagerly; `/health` reports
compiled graphs at readiness and now. See the [Apple Silicon recipe](../../../recipe/laya/apple-silicon.md).

```sh
LAYA_DEVICE=mps LAYA_MODELS=english python src/models/laya/worker.py
python -m pytest src/models/laya/tests                     # unit tests, no model
LAYA_CONTRACT=1 python -m pytest src/models/laya/tests     # plus contract tests on CPU, loads the checkpoint
```
