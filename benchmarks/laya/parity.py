"""Compare every run's answers with a reference run, using the tolerances declared in advance.

    python benchmarks/laya/parity.py benchmarks/laya/results/*.jsonl --ref C1

Per question: the decision must match (choice: chosen option; score: most likely level; noul: side of
0.5) and the largest absolute probability difference must stay within tolerance: 1e-3 when the request
ran in fp32, 1e-2 when it ran under fp16 autocast (MPS and rows >= the worker's amp threshold).
A flipped decision is reported with the reference margin between its top two outcomes.
Exits 1 when any question fails.
"""

import argparse
import json
import sys

TOLERANCE = {"fp32": 1e-3, "fp16": 1e-2}


def read(paths):
    envs, answers = {}, {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["config"], r["run"])
                if r["type"] == "env":
                    envs[key] = r
                elif r["type"] == "answers":
                    answers.setdefault(key, {})[r["workload"]] = r["answers"]
    return envs, answers


def outcome(answer):
    """(decision, {outcome: probability}) for one question's answer."""
    kind = answer["type"]
    if kind == "noul":
        p = answer["noul"]
        return p >= 0.5, {"yes": p}
    probs = answer["probabilities"]
    if kind == "choice":
        return answer["choice"], probs
    return max(probs, key=probs.get), probs  # score: most likely level


def margin(probs):
    if len(probs) == 1:  # noul: distance from the 0.5 boundary
        return abs(next(iter(probs.values())) - 0.5)
    top = sorted(probs.values(), reverse=True)
    return top[0] - top[1]


def path(env, rows):
    fp16 = (
        env.get("device_actual") == "mps"
        and env.get("amp_dtype") == "torch.float16"
        and rows >= (env.get("mps_amp_min_rows") or 10**9)
    )
    return "fp16" if fp16 else "fp32"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--ref", default="C1", help="reference config")
    parser.add_argument("--ref-run", help="reference run label (default: the first one found)")
    args = parser.parse_args()
    envs, answers = read(args.files)

    refs = sorted(k for k in answers if k[0] == args.ref and (not args.ref_run or k[1] == args.ref_run))
    if not refs:
        sys.exit(f"no answers for reference config {args.ref}")
    ref_key = refs[0]
    reference = answers[ref_key]

    print(f"Reference: {ref_key[0]} / {ref_key[1]}. Tolerances: fp32 {TOLERANCE['fp32']}, fp16 {TOLERANCE['fp16']}.\n")
    print(
        "| config | run | workload | question | type | path | decision ref → run | max abs Δp | ref margin | result |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    failed = total = 0
    for key in sorted(answers):
        if key == ref_key:
            continue
        env = envs.get(key, {})
        for workload, questions in sorted(reference.items()):
            got = answers[key].get(workload)
            if got is None:
                print(f"| {key[0]} | {key[1]} | {workload} | | | | missing | | | FAIL |")
                failed += 1
                total += 1
                continue
            precision = path(env, len(questions))
            for qid, ref_answer in sorted(questions.items()):
                total += 1
                ref_decision, ref_probs = outcome(ref_answer)
                decision, probs = outcome(got[qid])
                delta = max(abs(ref_probs.get(o, 0.0) - probs.get(o, 0.0)) for o in ref_probs.keys() | probs.keys())
                ok = decision == ref_decision and delta <= TOLERANCE[precision]
                failed += not ok
                flip = f"{ref_decision} → {decision}" if decision != ref_decision else f"{decision}"
                print(
                    f"| {key[0]} | {key[1]} | {workload} | {qid} | {ref_answer['type']} | {precision} | {flip} "
                    f"| {delta:.4f} | {margin(ref_probs):.4f} | {'PASS' if ok else 'FAIL'} |"
                )
    print(f"\n{total - failed}/{total} questions within tolerance.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
