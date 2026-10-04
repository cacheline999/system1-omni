"""Contract tests against a real worker process. They load the Laya checkpoint, so they only run with
LAYA_CONTRACT=1. The worker runs on the CPU unless LAYA_CONTRACT_DEVICE says otherwise; on an Apple Silicon
Mac the same contract is checked on the GPU, with and without the options:

    LAYA_CONTRACT=1 PYTHONPATH=src python -m pytest tests/laya/test_contract.py
    LAYA_CONTRACT=1 LAYA_CONTRACT_DEVICE=mps PYTHONPATH=src python -m pytest tests/laya/test_contract.py
    LAYA_CONTRACT=1 LAYA_CONTRACT_DEVICE=mps LAYA_CONTRACT_FLAGS="--compile --weights fp16" \
        PYTHONPATH=src python -m pytest tests/laya/test_contract.py
"""

import http.client
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LAYA_CONTRACT") != "1", reason="set LAYA_CONTRACT=1"
)

SRC = Path(__file__).resolve().parents[2] / "src"
DEVICE = os.environ.get("LAYA_CONTRACT_DEVICE", "cpu")
FLAGS = shlex.split(os.environ.get("LAYA_CONTRACT_FLAGS", ""))
CHECKPOINT = "convaiinnovations/laya"
TOKEN = "contract-test-token"
STATE = "I was charged twice for my order. Please refund the duplicate today."
CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges and refunds", "technical": "Software problems"},
}
SCORE = {
    "type": "score",
    "instructions": "How urgent is the request?",
    "criteria": ["Not urgent", "Needs attention soon", "Needs attention immediately"],
}
NOUL = {"type": "noul", "instructions": "Does the customer ask for a refund?"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(port, method, path, body=None, token=TOKEN, raw=False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = body if raw or body is None else json.dumps(body).encode()
    started = time.perf_counter()
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()
    payload = response.read()
    ms = (time.perf_counter() - started) * 1000
    conn.close()
    return response.status, payload, ms


def decide(port, questions, **kwargs):
    return call(
        port,
        "POST",
        "/v1/systemone",
        {"model": "english", "state": STATE, "questions": questions},
        **kwargs,
    )


@pytest.fixture(scope="module")
def worker():
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(SRC), "LAYA_API_KEY": TOKEN}
    command = [
        sys.executable,
        "-m",
        "frontend.laya_mps",
        "--device",
        DEVICE,
        "--require-device",
        "--model",
        "english",
        "--port",
        str(port),
        "--log-level",
        "warning",
        *FLAGS,
    ]
    process = subprocess.Popen(command, env=env)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        assert process.poll() is None, f"worker exited with {process.returncode}"
        try:
            status, body, _ = call(port, "GET", "/health")
            if status == 200:
                startup["first_health"] = json.loads(body)
                break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("worker not ready in 600 s")
    # The first decision after readiness, before any other test warms anything.
    first = decide(port, {"q": CHOICE})
    yield port, first
    process.terminate()
    process.wait(timeout=30)


startup = {}


@pytest.fixture(scope="module")
def reference():
    """Laya itself, in this process on the CPU in fp32, for every checkpoint: what the worker's answers
    are compared with."""
    from laya.router import Router

    return Router(device="cpu")


def test_health_reports_the_loaded_model(worker):
    port, _ = worker
    status, body, _ = call(port, "GET", "/health")
    health = json.loads(body)
    assert status == 200
    assert health["ready"] is True
    assert health["device"] == DEVICE
    assert health["device_mismatch"] is False
    assert health["checkpoint"] == CHECKPOINT
    assert health["loaded"] == ["english"]
    assert health["weights_dtype"] == (
        "torch.float16" if "fp16" in FLAGS else "torch.float32"
    )
    assert len(health["revision"]) == 40


def test_the_first_health_answer_comes_after_the_warmup(worker):
    # laya-serve also refuses connections while it loads, then answers /health before any forward pass;
    # the worker's first answer must already report a finished warmup.
    assert (
        startup["first_health"]["ready"] is True
        and startup["first_health"]["warmup_ms"] > 0
    )


def test_the_checkpoint_stays_loaded_across_requests(worker):
    port, _ = worker
    for _ in range(5):
        assert decide(port, {"q": NOUL})[0] == 200
    health = json.loads(call(port, "GET", "/health")[1])
    assert (
        health["loaded"] == ["english"]
        and health["warmup_ms"] == startup["first_health"]["warmup_ms"]
    )


def test_the_first_request_after_ready_succeeds(worker):
    # Its latency is measured by the benchmarks, not asserted here: on MPS it depends on
    # how long the GPU has been idle since the warmup (tens to over a hundred ms on an
    # M1 Pro), and on an M5 even a worker without warmup answered its first request in
    # 132-143 ms, so no bound separates warm from cold on every Mac. That the warmup ran
    # before ready is checked above.
    _, (status, body, _) = worker
    assert status == 200
    answer = json.loads(body)["answers"]["q"]
    assert answer["type"] == "choice" and answer["choice"] in CHOICE["criteria"]


@pytest.mark.parametrize(
    ("questions", "kinds"),
    [
        ({"q": CHOICE}, {"q": "choice"}),
        ({"q": SCORE}, {"q": "score"}),
        ({"q": NOUL}, {"q": "noul"}),
        (
            {"a": CHOICE, "b": SCORE, "c": NOUL},
            {"a": "choice", "b": "score", "c": "noul"},
        ),
    ],
    ids=["choice", "score", "noul", "combined"],
)
def test_decisions(worker, questions, kinds):
    port, _ = worker
    status, body, _ = decide(port, questions)
    assert status == 200
    result = json.loads(body)
    assert set(result["answers"]) == set(kinds)
    assert result["usage"]["input_tokens"] > 0
    for qid, kind in kinds.items():
        answer = result["answers"][qid]
        assert answer["type"] == kind
        if kind == "noul":
            assert 0.0 <= answer["noul"] <= 1.0
        else:
            assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        if kind == "choice":
            assert answer["choice"] in CHOICE["criteria"]


SIX = {f"q{i}": q for i, q in enumerate([CHOICE, SCORE, NOUL, CHOICE, SCORE, NOUL])}
QUESTIONS = {
    "choice": {"q": CHOICE},
    "score": {"q": SCORE},
    "noul": {"q": NOUL},
    "combined": {"a": CHOICE, "b": SCORE, "c": NOUL},
    "five-questions": {
        f"q{i}": q for i, q in enumerate([CHOICE, SCORE, NOUL] * 2) if i < 5
    },
    "six-questions": SIX,
    "eight-questions": {
        f"q{i}": q for i, q in enumerate([CHOICE, SCORE, NOUL] * 3) if i < 8
    },
}
LONG_STATE = " ".join(
    [
        "I ordered a laptop stand and a keyboard on the first of the month and paid by card.",
        "The stand arrived but the keyboard did not, and the tracking page has not changed in ten days.",
        "Yesterday I noticed two charges for the same order on my statement, one of them pending.",
        "I wrote to support twice and only received an automatic reply with a ticket number.",
        "I need the keyboard for work next week, otherwise I would rather cancel that part.",
    ]
    * 3
)


def decision(answer):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    return max(answer["probabilities"], key=answer["probabilities"].get)


@pytest.mark.parametrize("model", ["english", "multilingual"])
@pytest.mark.parametrize("state", [STATE, LONG_STATE], ids=["short", "long"])
@pytest.mark.parametrize("questions", list(QUESTIONS.values()), ids=list(QUESTIONS))
def test_answers_match_laya_itself(worker, reference, model, state, questions):
    port, _ = worker
    body = {"model": model, "state": state, "questions": questions}
    status, payload, _ = call(port, "POST", "/v1/systemone", body)
    assert status == 200
    served = json.loads(payload)
    expected = reference.predict(state, questions, model=model)
    assert set(expected) <= set(
        served
    )  # the worker adds nothing laya does not, and drops nothing
    assert served["usage"] == expected["usage"]
    reduced = "fp16" in FLAGS or (
        DEVICE == "mps"
        and len(questions) >= startup["first_health"]["mps_amp_min_rows"]
    )
    tolerance = 1e-2 if reduced else 1e-3
    worst, flipped = 0.0, []
    for qid, want in expected["answers"].items():
        got = served["answers"][qid]
        assert set(got) == set(want)
        if want["type"] == "noul":
            worst = max(worst, abs(got["noul"] - want["noul"]))
        else:
            assert set(got["probabilities"]) == set(want["probabilities"])
            worst = max(
                worst,
                *(
                    abs(got["probabilities"][o] - p)
                    for o, p in want["probabilities"].items()
                ),
            )
        if want["type"] == "choice":
            assert got["choice"] == decision(got)
        if decision(got) != decision(want):
            flipped.append(qid)
    print(f"max |dp| {worst:.4f}, flipped {flipped}")  # shown with pytest -rP
    assert worst <= tolerance and not flipped, (
        f"max |dp| {worst:.4f} (tolerance {tolerance}), flipped {flipped}"
    )


def test_same_request_same_answer(worker):
    port, _ = worker
    first, second = (
        json.loads(decide(port, {"a": CHOICE, "b": NOUL})[1])["answers"]
        for _ in range(2)
    )
    assert first == second


@pytest.mark.parametrize(
    ("body", "raw", "token", "expected"),
    [
        (b"{not json", True, TOKEN, 400),
        ({"model": "english", "state": STATE}, False, TOKEN, 400),
        (
            {
                "model": "english",
                "state": STATE,
                "questions": {"q": {"type": "bogus", "instructions": "?"}},
            },
            False,
            TOKEN,
            422,
        ),
        (b"x" * (2 * 1024 * 1024 + 1), True, TOKEN, 413),
        (
            {"model": "english", "state": STATE, "questions": {"q": NOUL}},
            False,
            "wrong",
            401,
        ),
        (
            {"model": "english", "state": STATE, "questions": {"q": NOUL}},
            False,
            None,
            401,
        ),
    ],
    ids=[
        "malformed-json",
        "no-questions",
        "bad-question",
        "too-large",
        "wrong-token",
        "no-token",
    ],
)
def test_errors(worker, body, raw, token, expected):
    port, _ = worker
    status, payload, _ = call(port, "POST", "/v1/systemone", body, token=token, raw=raw)
    assert status == expected, payload[:200]
