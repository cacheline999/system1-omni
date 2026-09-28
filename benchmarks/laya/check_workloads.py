"""T1 check: tokens per row of every workload, measured with Laya's own tokenizer (usage.input_tokens).

Each question is sent alone so its row length is exact; bench workloads must land within ±10% of target.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import laya

agent = laya.load("convaiinnovations/laya", device="cpu")
max_len = agent.cfg.get("max_len", 512)
failed = 0
with open(Path(__file__).resolve().parent / "workloads.jsonl") as f:
    workloads = [json.loads(line) for line in f]
for w in workloads:
    rows = [agent.system_one(w["state"], {qid: q})["usage"]["input_tokens"] for qid, q in w["questions"].items()]
    mean = sum(rows) / len(rows)
    target = w["target_tokens_per_row"]
    ok = target is None or abs(mean - target) <= 0.1 * target
    ok = ok and max(rows) < max_len  # a full row means Laya cut the state
    failed += not ok
    print(f"{'ok ' if ok else 'BAD'} {w['id']:5} rows={len(rows)} tokens/row={rows} mean={mean:.0f} target={target}")
sys.exit(1 if failed else 0)
