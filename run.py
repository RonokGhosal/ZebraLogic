"""Run the extraction-vs-reasoning experiment on both benchmarks via Ollama Qwen.

Loads a stratified subset of each benchmark, runs conditions A/B/C, scores
them, prints the decomposition + generation health (truncation), and saves all
raw outputs + scores to results.json.

  A = read the problem (prose/dialogue) -> answer
  B = read the clean formal facts        -> answer   (A->B gap = "reading" cost)
  C = extract the formal facts           -> vs gold  (which facts get misread)

Usage (after `ollama serve` is up and qwen3:14b is pulled):
    python run.py --n 2                 # small smoke run
    python run.py --n 10                # bigger
    python run.py --mock                # no model: validates the pipeline
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


def _gen(model):
    """(text, meta) generator that tolerates MockModel (no metadata)."""
    f = getattr(model, "generate_with_meta", None)
    return f or (lambda p, label="": (model.generate(p), {}))


def _per_instance(insts, model, one, bench, progress=None, save=None):
    """Run `one(inst)` per instance (threaded by model.workers), scoring and
    saving incrementally so dashboards/results are live and crash-safe."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows: list = [None] * len(insts)
    lock, done = threading.Lock(), 0

    def task(k):
        t0 = time.time()
        return k, one(insts[k]), time.time() - t0

    with ThreadPoolExecutor(max_workers=getattr(model, "workers", 1)) as ex:
        for fut in as_completed([ex.submit(task, k) for k in range(len(insts))]):
            k, row, secs = fut.result()
            with lock:
                rows[k] = row
                done += 1
                if progress:
                    slim = {x: v for x, v in row.items() if not x.startswith("raw")}
                    progress.item_done(bench, slim, secs)
                if save:
                    save([r for r in rows if r is not None])
            print(f"\r    {bench} {done}/{len(insts)} items", end="", flush=True)
    sys.stdout.write("\n")
    return [r for r in rows if r is not None]


def run_zebra(insts, model, progress=None, save=None):
    gen = _gen(model)

    def one(i):
        a, ma = gen(Z.prompt_A(i), f"zebra-A {i['id']}")
        b, mb = gen(Z.prompt_B(i), f"zebra-B {i['id']}")
        c, mc = gen(Z.prompt_C(i), f"zebra-C {i['id']}")
        sa, sb, sc = Z.score_answer(a, i), Z.score_answer(b, i), Z.score_extraction(c, i)
        return {"id": i["id"], "size": i["size"], "A_full": sa["full"], "A_cell": sa["cell"],
                "B_full": sb["full"], "B_cell": sb["cell"], "C_recall": sc["recall"],
                "A_trunc": bool(ma.get("truncated")), "B_trunc": bool(mb.get("truncated")),
                "C_trunc": bool(mc.get("truncated")),
                "A_tokens": ma.get("output_tokens", 0), "B_tokens": mb.get("output_tokens", 0),
                "C_tokens": mc.get("output_tokens", 0),
                "rawA": a, "rawB": b, "rawC": c}

    return _per_instance(insts, model, one, "zebra", progress, save)


def run_tcp(insts, model, progress=None, save=None):
    gen = _gen(model)

    def one(i):
        a, ma = gen(TC.prompt_A(i), f"tcp-A {i['id']}")
        b, mb = gen(TC.prompt_B(i), f"tcp-B {i['id']}")
        c, mc = gen(TC.prompt_C(i), f"tcp-C {i['id']}")
        return {"id": i["id"], "regime": i["regime"],
                "A": TC.score_answer(a, i["gold"], i["regime"]),
                "B": TC.score_answer(b, i["gold"], i["regime"]),
                "C": TC.score_extraction(c, i["spec"]),
                "A_trunc": bool(ma.get("truncated")), "B_trunc": bool(mb.get("truncated")),
                "C_trunc": bool(mc.get("truncated")),
                "rawA": a, "rawB": b, "rawC": c}

    return _per_instance(insts, model, one, "tcp", progress, save)


def truncation_summary(model):
    calls = getattr(model, "calls", None)
    if not calls:
        return "  (no call metadata — mock run)"
    trunc = sum(1 for c in calls if c.get("truncated"))
    errs = sum(1 for c in calls if c.get("error"))
    toks = [c.get("output_tokens", 0) for c in calls]
    return (
        f"  calls: {len(calls)} | truncated: {trunc} | errors: {errs} | "
        f"output tokens avg {sum(toks)//max(len(toks),1)}, max {max(toks, default=0)}"
        + ("\n  WARNING: truncated calls score as wrong — raise --max-tokens before trusting these numbers" if trunc else "")
    )


def make_model(args, progress=None):
    from zebralogic.model_client import OllamaModel

    return OllamaModel(
        base_url=args.base_url, model=args.model, max_tokens=args.max_tokens,
        num_ctx=args.num_ctx, seed=args.seed, workers=args.workers, progress=progress,
    )


def make_progress(args, mode: str):
    from zebralogic.progress import Progress

    # mock/test runs must not clobber the live dashboard watched by monitor agents
    root = "/tmp/zebra-mock-dashboard" if getattr(args, "mock", False) else "."
    return Progress(config={"mode": mode, **{k: v for k, v in vars(args).items() if v is not None}},
                    root=root)


def run_referee_mode(args):
    from zebralogic import referee as R

    progress = make_progress(args, "referee")
    print("loading ZebraLogic (parsing + solving gold)...", flush=True)
    zb = stratified(Z.load_zebra(), lambda d: d["size"], args.n)
    if args.limit:
        zb = zb[: args.limit]
    print(f"  referee on {len(zb)} puzzles (max_retries={args.retries})", flush=True)
    model = make_model(args, progress)
    progress.add_total("referee", len(zb))
    progress.set_phase("referee")
    save = lambda rows: json.dump(rows, open("results_referee.json", "w"), indent=1)  # noqa: E731
    t0 = time.time()
    rows = R.compare(zb, model, max_retries=args.retries, workers=model.workers,
                     progress=progress, save=save)
    progress.finish(ok=True)
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
    print(truncation_summary(model))
    json.dump(rows, open("results_referee.json", "w"), indent=1)
    print("\nsaved results_referee.json")


def run_origin_mode(args):
    from zebralogic import origin_experiment as O

    progress = make_progress(args, "origin")
    print("loading ZebraLogic (parsing + solving gold)...", flush=True)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    zb = stratified([i for i in Z.load_zebra() if i["size"] in sizes],
                    lambda d: d["size"], args.n)
    if args.limit:
        zb = zb[: args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"  origin on {len(zb)} puzzles (sizes {sizes}, arms {arms}, "
          f"max_retries={args.retries})", flush=True)
    model = Z.MockModel(zb, mode="lossy") if args.mock else make_model(args, progress)
    progress.add_total("origin", len(zb))
    progress.set_phase("origin")
    save = lambda rows: json.dump(rows, open("results_origin.json", "w"), indent=1)  # noqa: E731
    t0 = time.time()
    rows = O.run_origin(zb, model, arms, max_retries=args.retries,
                        workers=getattr(model, "workers", 1), progress=progress, save=save)
    progress.finish(ok=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    print(O.summarize(rows, arms))
    print(truncation_summary(model))
    json.dump(rows, open("results_origin.json", "w"), indent=1)
    print("\nsaved results_origin.json (includes all raw outputs)")


def run_closer_mode(args):
    from zebralogic import origin_experiment as O

    progress = make_progress(args, "closer")
    rows = json.load(open(args.from_results))
    fails = [r for r in rows if r.get("status2") == "failure" and "misread2" in r]
    print(f"  closer on {len(fails)} failures from {args.from_results} "
          f"(max_retries={args.retries})", flush=True)
    print("loading ZebraLogic (parsing + solving gold)...", flush=True)
    insts = {i["id"]: i for i in Z.load_zebra()}
    missing = [r["id"] for r in fails if r["id"] not in insts]
    if missing:
        raise SystemExit(f"failure ids not found in loader output: {missing}")
    model = make_model(args, progress)
    progress.add_total("closer", len(fails))
    progress.set_phase("closer")
    save = lambda rs: json.dump(rs, open("results_closer.json", "w"), indent=1)  # noqa: E731
    t0 = time.time()
    out = O.run_closer(fails, insts, model, max_retries=args.retries,
                       workers=model.workers, progress=progress, save=save)
    progress.finish(ok=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    print(O.summarize_closer(out))
    print(truncation_summary(model))
    json.dump(out, open("results_closer.json", "w"), indent=1)
    print("\nsaved results_closer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="scale: zebra=n/size, tcp=10n/regime")
    ap.add_argument("--limit", type=int, default=None, help="hard cap on instances per benchmark (fast first pass)")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="num_predict: thinking+answer budget (Qwen3 eval protocol uses 32k)")
    ap.add_argument("--num-ctx", type=int, default=40960,
                    help="context window; must cover prompt + max-tokens or Ollama silently truncates")
    ap.add_argument("--seed", type=int, default=1234, help="sampling seed (temp 0.6 per Qwen3 card; no greedy)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (1: a 36GB Mac fits one 40k-ctx slot)")
    ap.add_argument("--bench", choices=["zebra", "tcp", "both"], default="both",
                    help="which benchmark(s) to run in A/B/C mode")
    ap.add_argument("--out", default="results.json", help="output file for A/B/C results")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--referee", action="store_true", help="run the referee (raw vs refereed ZebraLogic) instead of A/B/C")
    ap.add_argument("--retries", type=int, default=3, help="referee/origin: max retries per puzzle")
    ap.add_argument("--origin", action="store_true",
                    help="run the error-origin experiment (misread vs reasoning failures; use --n ~20)")
    ap.add_argument("--sizes", default="5*5,5*6,6*4,6*5,6*6",
                    help="origin: comma-separated sizes to harvest failures from")
    ap.add_argument("--arms", default="binary,location,interp",
                    help="origin: feedback arms to run from each failure")
    ap.add_argument("--closer", action="store_true",
                    help="run the closing experiment (self-referee + varied-seed) on the origin run's failures")
    ap.add_argument("--from-results", default="results_origin_regraded.json",
                    help="closer: regraded origin results to take failures from")
    args = ap.parse_args()

    if args.closer:
        run_closer_mode(args)
        return
    if args.origin:
        run_origin_mode(args)
        return
    if args.referee:
        run_referee_mode(args)
        return

    print("loading benchmarks (parsing + solving gold)...", flush=True)
    zb = stratified(Z.load_zebra(), lambda d: d["size"], args.n) if args.bench in ("zebra", "both") else []
    tc = stratified(TC.load_tcp(), lambda d: d["regime"], args.n * 10) if args.bench in ("tcp", "both") else []
    if args.limit:
        zb, tc = zb[: args.limit], tc[: args.limit]
    print(f"  ZebraLogic: {len(zb)} instances | TCP: {len(tc)} instances", flush=True)

    progress = make_progress(args, "mock-abc" if args.mock else "abc")
    if args.mock:
        zmodel, tmodel = Z.MockModel(zb), TC.MockModel(tc)
    else:
        zmodel = tmodel = make_model(args, progress)
    if zb:
        progress.add_total("zebra", len(zb))
    if tc:
        progress.add_total("tcp", len(tc))

    results = {"zebra": [], "tcp": []}

    def saver(bench):
        def f(rows):
            results[bench] = rows
            json.dump(results, open(args.out, "w"), indent=1)
        return f

    t0 = time.time()
    zrows, trows = [], []
    if zb:
        print("running ZebraLogic...", flush=True)
        progress.set_phase("zebra A/B/C")
        zrows = run_zebra(zb, zmodel, progress, saver("zebra"))
        print(f"  ZebraLogic done ({time.time()-t0:.0f}s)", flush=True)
    if tc:
        print("running TCP...", flush=True)
        progress.set_phase("tcp A/B/C")
        trows = run_tcp(tc, tmodel, progress, saver("tcp"))
    progress.finish(ok=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    # ----- decomposition summary -----
    print("\n================  RESULTS  ================")
    if zrows:
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

    if trows:
        print("\nTCP (exact-answer accuracy; C=per-field extraction):")
        print(f"  A read-dialogue: {mean(r['A'] for r in trows)}")
        print(f"  B clean-facts  : {mean(r['B'] for r in trows)}")
        cf = collections.defaultdict(list)
        for r in trows:
            for f, ok in r["C"].items():
                cf[f].append(bool(ok))
        print("  C extraction by field: " + ", ".join(f"{f}={mean(v)}" for f, v in cf.items()))

    print("\nGeneration health:")
    print(truncation_summary(zmodel))

    json.dump({"zebra": zrows, "tcp": trows}, open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out} (includes raw model outputs)")


if __name__ == "__main__":
    main()
