"""Laya worker: laya-serve with warmup before readiness and the actual device in /health.

laya-serve (laya 0.3.20) answers /health as soon as it binds, before any forward pass, and reports
LAYA_DEVICE as configured rather than where the model ended up. This worker reuses laya's app and
request handling unchanged and fixes both:

- It binds only after a warmup that covers short, long and multi-question requests (the last one
  crosses laya's fp16 autocast threshold on MPS), so a reachable worker is a warm one.
- /health reports the device, weight and autocast dtypes of the loaded agent, the checkpoint and
  revision it serves, and whether the device differs from the one requested.

Configuration is laya-serve's (LAYA_HOST, LAYA_PORT, LAYA_DEVICE, LAYA_MODELS, LAYA_API_KEY, ...) plus:

    LAYA_WORKER_MODEL          model to warm up and describe          english
    LAYA_REQUIRE_DEVICE        exit instead of serving on another     0
                               device than LAYA_DEVICE asked for
    LAYA_WORKER_COMPILE        off, all, or single: torch.compile      off
                               (dynamic=True) the model before warmup,
                               for every batch or one-row batches only

The warmup also compiles every shape class it sends through the compiled model: in measured runs on an
M1 Pro the worker with `single` was ready after about 22 s instead of 8 s (`all` took over a minute).
/health counts compiled graphs at readiness and now; `recompiled_after_ready` means a request hit a
shape class the warmup did not cover. `single` exists because on MPS compiling cut one-question
latency by about 29% while compiling everything made multi-question requests slower (benchmarks/laya).
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("laya-worker")

# (words of state, questions): each shape runs twice. Short, mid-length and near-window states, then
# 3 and 6 questions (6 is at or above laya's MPS fp16 autocast threshold of 5 rows).
_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges and refunds", "technical": "Software problems", "other": "Anything else"},
}
_SCORE = {"type": "score", "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]}
_NOUL = {"type": "noul", "instructions": "Does the customer ask for a refund?"}
WARMUP_SHAPES = [
    (10, {"q": _CHOICE}),
    (150, {"q": _CHOICE}),
    (400, {"q": _CHOICE}),
    (10, {"a": _CHOICE, "b": _SCORE, "c": _NOUL}),
    (10, {f"q{i}": q for i, q in enumerate([_CHOICE, _SCORE, _NOUL, _CHOICE, _SCORE, _NOUL])}),
]
WARMUP_REPEATS = 2


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def warmup(router: Any, model: str, shapes=WARMUP_SHAPES, repeats: int = WARMUP_REPEATS) -> dict[str, Any]:
    """Run every shape `repeats` times. Any failure propagates: a worker that cannot answer must not bind."""
    started = time.perf_counter()
    routing = None
    for words, questions in shapes:
        state = " ".join(["refund"] * words)
        for _ in range(repeats):
            result = router.predict(state, questions, model=model)
            routing = result.get("routing") or routing
    return {"warmup_ms": round((time.perf_counter() - started) * 1000, 1), "routing": routing}


def compiled_graphs() -> int:
    """Graphs torch.compile has produced in this process."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


COMPILE_MODES = {"0": "off", "off": "off", "": "off", "1": "all", "all": "all", "single": "single"}


def compile_agent(agent: Any, mode: str) -> None:
    """`all`: every forward goes through the compiled model. `single`: batches of one row (one question)
    do, everything else runs eager. Both paths share the same parameters."""
    import torch

    eager = agent.model
    compiled = torch.compile(eager, dynamic=True)
    if mode == "all":
        agent.model = compiled
        return

    class SingleRowCompiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.eager = eager

        def forward(self, input_ids, *args, **kwargs):
            model = compiled if input_ids.shape[0] == 1 else self.eager
            return model(input_ids, *args, **kwargs)

    agent.model = SingleRowCompiled()


def _cached_revision(repo: str | None, ref: str = "main") -> str | None:
    if not repo:
        return None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return (Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / ref).read_text().strip()
    except (ImportError, OSError):
        return None


def describe(agent: Any, requested: str | None, routing: dict[str, Any] | None) -> dict[str, Any]:
    """What /health reports about the loaded agent."""
    device = str(getattr(agent, "device", "unknown"))
    model = getattr(agent, "model", None)
    try:
        weights = str(next(model.parameters()).dtype) if model is not None else None
    except (AttributeError, StopIteration, TypeError):
        weights = None
    repo = (routing or {}).get("repo")
    requested_type = requested.split(":")[0] if requested else None
    return {
        "device": device,
        "requested_device": requested or "auto",
        "device_mismatch": bool(requested_type) and device.split(":")[0] != requested_type,
        "weights_dtype": weights,
        "autocast_dtype": str(getattr(agent, "dtype", None)),
        "mps_amp_min_rows": getattr(agent, "mps_amp_min_rows", None),
        "checkpoint": repo,
        "revision": _cached_revision(repo),
    }


def create_worker_app(
    router: Any,
    model: str,
    requested: str | None,
    require_device: bool = False,
    compile: str = "off",
    graph_counter=compiled_graphs,
):
    """Optionally compile, warm the router up, then return laya's app with /health replaced.
    Raises if compiling or warmup fails."""
    from laya.serve import create_app

    if compile not in ("off", "all", "single"):
        raise ValueError(f"compile mode must be off, all or single, not {compile!r}")
    if compile != "off":
        compile_agent(router.load(model), compile)
    warm = warmup(router, model)
    info = describe(router.load(model), requested, warm["routing"])
    info["warmup_ms"] = warm["warmup_ms"]
    graphs_at_ready = graph_counter() if compile != "off" else None
    if info["device_mismatch"]:
        message = f"asked for {info['requested_device']}, model is on {info['device']}"
        if require_device:
            raise RuntimeError(message)
        log.warning(message)

    app = create_app(router)
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        compiled = {"mode": compile}
        if compile != "off":
            now = graph_counter()
            compiled.update(
                graphs_at_ready=graphs_at_ready, graphs_now=now, recompiled_after_ready=now > graphs_at_ready
            )
        return {"status": "ok", "ready": True, "loaded": router.loaded, **info, "compile": compiled}

    return app


def main() -> None:
    import uvicorn
    from laya.serve import _resolve_port, build_router

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    model = os.environ.get("LAYA_WORKER_MODEL", "english")
    requested = os.environ.get("LAYA_DEVICE") or None
    try:
        app = create_worker_app(
            build_router(),
            model,
            requested,
            require_device=_env_bool("LAYA_REQUIRE_DEVICE"),
            compile=COMPILE_MODES.get(os.environ.get("LAYA_WORKER_COMPILE", "").strip().lower(), "invalid"),
        )
    except Exception as exc:  # noqa: BLE001 -- any failure before binding means not ready, ever
        sys.exit(f"laya-worker: not starting: {exc}")
    uvicorn.run(
        app,
        host=os.environ.get("LAYA_HOST", "127.0.0.1"),
        port=_resolve_port(),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
