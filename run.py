"""Run the extraction-vs-reasoning experiment on both benchmarks via vLLM Qwen.

Loads a stratified subset of each benchmark, runs conditions A/B/C with batched
(concurrent) model calls, scores them, prints the decomposition, and saves all
raw outputs + scores to results.json.

  A = read the problem (prose/dialogue) -> answer
  B = read the clean formal facts        -> answer   (A->B gap = "reading" cost)
  C = extract the formal facts           -> vs gold  (which facts get misread)

Usage (on the GPU box, after vLLM is up):
    python run.py --n 2                 # small smoke run
    python run.py --n 10                # bigger
    python run.py --mock                # no GPU: validates the pipeline
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time

sys.path.insert(0, "src")

from zebralogic import tcp_experiment as TC  # noqa: E402
from zebralogic import zebra_experiment as Z  # noqa: E402


def stratified(items, keyf, per):
    by = collections.defaultdict(list)
    for it in items:
        by[keyf(it)].append(it)
    return [x for k in sorted(by) for x in by[k][:per]]


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def run_zebra(insts, model):
    rA = model.generate_many([Z.prompt_A(i) for i in insts], "zebra-A")
    rB = model.generate_many([Z.prompt_B(i) for i in insts], "zebra-B")
    rC = model.generate_many([Z.prompt_C(i) for i in insts], "zebra-C")
    rows = []
    for i, a, b, c in zip(insts, rA, rB, rC):
        sa, sb, sc = Z.score_answer(a, i), Z.score_answer(b, i), Z.score_extraction(c, i)
        rows.append(
            {"id": i["id"], "size": i["size"], "A_full": sa["full"], "A_cell": sa["cell"],
             "B_full": sb["full"], "B_cell": sb["cell"], "C_recall": sc["recall"],
             "rawA": a, "rawB": b, "rawC": c}
        )
    return rows


def run_tcp(insts, model):
    rA = model.generate_many([TC.prompt_A(i) for i in insts], "tcp-A")
    rB = model.generate_many([TC.prompt_B(i) for i in insts], "tcp-B")
    rC = model.generate_many([TC.prompt_C(i) for i in insts], "tcp-C")
    rows = []
    for i, a, b, c in zip(insts, rA, rB, rC):
        rows.append(
            {"id": i["id"], "regime": i["regime"],
             "A": TC.score_answer(a, i["gold"], i["regime"]),
             "B": TC.score_answer(b, i["gold"], i["regime"]),
             "C": TC.score_extraction(c, i["spec"]), "rawA": a, "rawB": b, "rawC": c}
        )
    return rows


def run_referee_mode(args):
    from zebralogic import referee as R
    from zebralogic.model_client import VLLMModel

    print("loading ZebraLogic (parsing + solving gold)...", flush=True)
    zb = stratified(Z.load_zebra(), lambda d: d["size"], args.n)
    if args.limit:
        zb = zb[: args.limit]
    print(f"  referee on {len(zb)} puzzles (max_retries={args.retries})", flush=True)
    model = VLLMModel(base_url=args.base_url, model=args.model, max_tokens=args.max_tokens)
    t0 = time.time()
    rows = R.compare(zb, model, max_retries=args.retries, workers=model.workers)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    raw, ref = mean(r["raw_full"] for r in rows), mean(r["ref_full"] for r in rows)
    print("\n=========  REFEREE (ZebraLogic)  =========")
    print(f"  raw Qwen  : {raw}")
    print(f"  refereed  : {ref}   (lift {ref - raw:+.3f})")
    print(f"  converged : {mean(r['converged'] for r in rows)} | avg attempts {mean(r['attempts'] for r in rows)}")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["size"]].append(r)
    print("  by size (raw -> ref):")
    for s in sorted(by):
        g = by[s]
        print(f"    {s}: {mean(x['raw_full'] for x in g)} -> {mean(x['ref_full'] for x in g)}")
    json.dump(rows, open("results_referee.json", "w"), indent=1)
    print("\nsaved results_referee.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="scale: zebra=n/size, tcp=10n/regime")
    ap.add_argument("--limit", type=int, default=None, help="hard cap on instances per benchmark (fast first pass)")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=8000, help="answer budget (raise if answers come back empty)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--referee", action="store_true", help="run the referee (raw vs refereed ZebraLogic) instead of A/B/C")
    ap.add_argument("--retries", type=int, default=3, help="referee: max retries per puzzle")
    args = ap.parse_args()

    if args.referee:
        run_referee_mode(args)
        return

    print("loading benchmarks (parsing + solving gold)...", flush=True)
    zb = stratified(Z.load_zebra(), lambda d: d["size"], args.n)
    tc = stratified(TC.load_tcp(), lambda d: d["regime"], args.n * 10)
    if args.limit:
        zb, tc = zb[: args.limit], tc[: args.limit]
    print(f"  ZebraLogic: {len(zb)} instances | TCP: {len(tc)} instances", flush=True)

    if args.mock:
        zmodel, tmodel = Z.MockModel(zb), TC.MockModel(tc)
    else:
        from zebralogic.model_client import VLLMModel
        zmodel = tmodel = VLLMModel(base_url=args.base_url, model=args.model, max_tokens=args.max_tokens)

    t0 = time.time()
    print("running ZebraLogic...", flush=True)
    zrows = run_zebra(zb, zmodel)
    json.dump({"zebra": zrows, "tcp": []}, open("results.json", "w"), indent=1)  # partial save
    print(f"  ZebraLogic done ({time.time()-t0:.0f}s); running TCP...", flush=True)
    trows = run_tcp(tc, tmodel)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    # ----- decomposition summary -----
    print("\n================  RESULTS  ================")
    print("ZebraLogic (puzzle-level accuracy; C=extraction recall):")
    print(f"  A read-puzzle : {mean(r['A_full'] for r in zrows)}  (cell {mean(r['A_cell'] for r in zrows)})")
    print(f"  B clean-facts : {mean(r['B_full'] for r in zrows)}  (cell {mean(r['B_cell'] for r in zrows)})")
    print(f"  C extraction  : {mean(r['C_recall'] for r in zrows)}")
    by = collections.defaultdict(list)
    for r in zrows:
        by[r["size"]].append(r)
    print("  by size (A_full / B_full / C_recall):")
    for s in sorted(by):
        g = by[s]
        print(f"    {s}: {mean(x['A_full'] for x in g)} / {mean(x['B_full'] for x in g)} / {mean(x['C_recall'] for x in g)}")

    print("\nTCP (exact-answer accuracy; C=per-field extraction):")
    print(f"  A read-dialogue: {mean(r['A'] for r in trows)}")
    print(f"  B clean-facts  : {mean(r['B'] for r in trows)}")
    cf = collections.defaultdict(list)
    for r in trows:
        for f, ok in r["C"].items():
            cf[f].append(bool(ok))
    print("  C extraction by field: " + ", ".join(f"{f}={mean(v)}" for f, v in cf.items()))

    json.dump({"zebra": zrows, "tcp": trows}, open("results.json", "w"), indent=1)
    print("\nsaved results.json (includes raw model outputs)")


if __name__ == "__main__":
    main()
