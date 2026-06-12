"""Post-hoc regrade of results_origin.json with the fixed (category-aware)
oracle and the format-tolerant reading grader.

Why this exists: the first origin run executed with a value-only position
lookup, which mis-evaluates constraints on the 21/774 puzzles where one value
lives in two categories ('alice' as Name AND Children, 'red' as Color AND
HairColor). For those instances the LIVE feedback sent to the model may have
named the wrong (or a satisfied) clue, so their repair-arm outcomes cannot be
regraded — only excluded. Everything else is recomputed from the saved raws.

Also measures retry lock-in: with a fixed sampling seed, once the model
re-emits the same grid the retry loop is deterministic — further attempts are
no-ops, so "didn't converge" can mean "locked in", not "tried and failed".

Usage:  python3 regrade_origin.py [results_origin.json]
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from zebralogic.origin_experiment import (  # noqa: E402
    clue_map,
    fisher_one_sided,
    grade_read,
    parse_localize,
)
from zebralogic.referee import cat_alias, violations  # noqa: E402
from zebralogic.zebra_experiment import _parse_grid, load_zebra, score_answer  # noqa: E402
from zebralogic.zebra_parser import build  # noqa: E402


def colliding_values(inst: dict) -> set[str]:
    seen: dict[str, str] = {}
    dup = set()
    for cat, vs in inst["categories"].items():
        for v in vs:
            k = str(v).strip().lower()
            if k in seen and seen[k] != cat:
                dup.add(k)
            seen[k] = cat
    return dup


def tainted(inst: dict, constraints) -> bool:
    """Live feedback unreliable: a colliding value sits inside a constraint."""
    dup = colliding_values(inst)
    return bool(dup) and any(
        str(c.a[1]).strip().lower() in dup or str(c.b[1]).strip().lower() in dup
        for c in constraints
    )


def main(path: str = "results_origin.json") -> None:
    rows = json.load(open(path))
    insts = {i["id"]: i for i in load_zebra()}
    print(f"regrading {len(rows)} rows from {path}")

    out = []
    for r in rows:
        inst = insts.get(r["id"])
        if inst is None:
            print(f"  SKIP {r['id']}: not in loader output")
            continue
        cons = build(inst["nl"])[0].constraints
        alias = cat_alias(inst)
        by_no, by_con = clue_map(inst["nl"])
        g = dict(r)
        g["tainted"] = tainted(inst, cons)

        # recompute status from the first answer with the fixed oracle
        grid0 = _parse_grid(r["raw0"])
        full0 = score_answer(r["raw0"], inst)["full"]
        if full0:
            g["status2"] = "correct"
        elif grid0 is None:
            g["status2"] = "no_parse"
        else:
            bad0 = violations(cons, grid0, alias)
            g["status2"] = "failure" if bad0 else "anomaly"
            g["violated_clues2"] = sorted({by_con[c] for c in bad0 if c in by_con})

        # regrade probes from saved raws (probe TARGETS were chosen live, so a
        # changed violation set on tainted rows also invalidates classification)
        if r.get("reads"):
            reads2 = [grade_read(p["raw"], by_no[p["clue_no"]][1])["ok"]
                      for p in r["reads"]]
            g["misread2"] = not all(reads2)
        if r.get("localize"):
            pred = parse_localize(r["localize"]["raw"])
            oracle = g.get("violated_clues2", r["localize"]["oracle"])
            g["localize2"] = {"hit": bool(pred and set(pred) & set(oracle)),
                              "exact": pred == oracle}

        # regrade arm convergence + lock-in from saved texts
        for arm, a in (r.get("arms") or {}).items():
            grids = [_parse_grid(t) for t in a["texts"]]
            conv = False
            for gr in grids:
                if gr is not None and not violations(cons, gr, alias):
                    conv = True
                    break
            a["converged2"] = conv
            seq = [_parse_grid(r["raw0"])] + grids
            a["locked_in"] = any(x is not None and x == y
                                 for x, y in zip(seq, seq[1:]))
        out.append(g)

    # ---------------- corrected summary ----------------
    def block(rows_, title):
        fails = [r for r in rows_ if r["status2"] == "failure" and "misread2" in r]
        mis = [r for r in fails if r["misread2"]]
        rea = [r for r in fails if not r["misread2"]]
        print(f"\n===== {title} (n={len(rows_)}) =====")
        for s in ("correct", "no_parse", "anomaly", "failure"):
            k = sum(1 for r in rows_ if r["status2"] == s)
            print(f"  {s}: {k}", end="")
        print(f"\n  failures by origin: misread {len(mis)} | reasoning {len(rea)}")
        loc = [r["localize2"] for r in fails if r.get("localize2")]
        if loc:
            print(f"  self-localization: hit {sum(x['hit'] for x in loc)}/{len(loc)}"
                  f" | exact {sum(x['exact'] for x in loc)}/{len(loc)}")
        arms = sorted({a for r in fails for a in (r.get("arms") or {})})
        for arm in arms:
            def rate(grp):
                g_ = [r for r in grp if arm in (r.get("arms") or {})]
                return sum(r["arms"][arm]["converged2"] for r in g_), len(g_)
            km, nm = rate(mis)
            kr, nr = rate(rea)
            p = fisher_one_sided(km, nm - km, kr, nr - kr)
            lock = sum(r["arms"][arm]["locked_in"] for r in fails
                       if arm in (r.get("arms") or {})
                       and not r["arms"][arm]["converged2"])
            tot = sum(1 for r in fails if arm in (r.get("arms") or {})
                      and not r["arms"][arm]["converged2"])
            print(f"  arm {arm:9}: repaired misread {km}/{nm} | reasoning {kr}/{nr}"
                  f" | p={p:.3f} | locked-in {lock}/{tot} unconverged")

    nt = [r for r in out if not r["tainted"]]
    block(out, "ALL (tainted arms regraded but NOT trustworthy)")
    block(nt, "CLEAN ONLY (collision-tainted excluded — primary analysis)")
    print(f"\n  tainted instances excluded above: "
          f"{sorted(r['id'] for r in out if r['tainted'])}")
    flips = [r["id"] for r in out
             if r.get("status2") != r.get("status")
             or any(a.get("converged2") != a.get("converged")
                    for a in (r.get("arms") or {}).values())]
    print(f"  rows where regrade changed status/convergence: {sorted(flips)}")
    json.dump(out, open("results_origin_regraded.json", "w"), indent=1)
    print("\nsaved results_origin_regraded.json")


if __name__ == "__main__":
    main(*sys.argv[1:2] or ["results_origin.json"])
