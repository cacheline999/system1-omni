"""Contract tests against a real worker process on CPU. They load the Laya checkpoint, so they only run
with LAYA_CONTRACT=1:

    LAYA_CONTRACT=1 python -m pytest src/models/laya/tests/test_contract.py
"""

import http.client
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("LAYA_CONTRACT") != "1", reason="set LAYA_CONTRACT=1")

WORKER = Path(__file__).resolve().parents[1] / "worker.py"
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
    return call(port, "POST", "/v1/systemone", {"model": "english", "state": STATE, "questions": questions}, **kwargs)


@pytest.fixture(scope="module")
def worker():
    port = free_port()
    env = {
        **os.environ,
        "LAYA_HOST": "127.0.0.1",
        "LAYA_PORT": str(port),
        "LAYA_DEVICE": "cpu",
        "LAYA_MODELS": "english",
        "LAYA_API_KEY": TOKEN,
        "LAYA_LOG_LEVEL": "warning",
    }
    process = subprocess.Popen([sys.executable, str(WORKER)], env=env)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        assert process.poll() is None, f"worker exited with {process.returncode}"
        try:
            status, _, _ = call(port, "GET", "/health")
            if status == 200:
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


def test_health_reports_the_loaded_model(worker):
    port, _ = worker
    status, body, _ = call(port, "GET", "/health")
    health = json.loads(body)
    assert status == 200
    assert health["ready"] is True
    assert health["device"] == "cpu"
    assert health["device_mismatch"] is False
    assert health["checkpoint"] == "convaiinnovations/laya"
    assert health["loaded"] == ["english"]


def test_first_request_after_ready_is_warm(worker):
    port, (status, _, first_ms) = worker
    assert status == 200
    warm = [decide(port, {"q": CHOICE})[2] for _ in range(20)]
    assert first_ms <= 2 * statistics.median(warm), f"first {first_ms:.0f} ms, warm p50 {statistics.median(warm):.0f}"


@pytest.mark.parametrize(
    ("questions", "kinds"),
    [
        ({"q": CHOICE}, {"q": "choice"}),
        ({"q": SCORE}, {"q": "score"}),
        ({"q": NOUL}, {"q": "noul"}),
        ({"a": CHOICE, "b": SCORE, "c": NOUL}, {"a": "choice", "b": "score", "c": "noul"}),
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


def test_same_request_same_answer(worker):
    port, _ = worker
    first, second = (json.loads(decide(port, {"a": CHOICE, "b": NOUL})[1])["answers"] for _ in range(2))
    assert first == second


@pytest.mark.parametrize(
    ("body", "raw", "token", "expected"),
    [
        (b"{not json", True, TOKEN, 400),
        ({"model": "english", "state": STATE}, False, TOKEN, 400),
        (
            {"model": "english", "state": STATE, "questions": {"q": {"type": "bogus", "instructions": "?"}}},
            False,
            TOKEN,
            422,
        ),
        (b"x" * (2 * 1024 * 1024 + 1), True, TOKEN, 413),
        ({"model": "english", "state": STATE, "questions": {"q": NOUL}}, False, "wrong", 401),
        ({"model": "english", "state": STATE, "questions": {"q": NOUL}}, False, None, 401),
    ],
    ids=["malformed-json", "no-questions", "bad-question", "too-large", "wrong-token", "no-token"],
)
def test_errors(worker, body, raw, token, expected):
    port, _ = worker
    status, payload, _ = call(port, "POST", "/v1/systemone", body, token=token, raw=raw)
    assert status == expected, payload[:200]
