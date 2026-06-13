"""Origin experiment: are errors born from MISREADING a clue immune to feedback?

Hypothesis: when a frozen model's grid violates a clue, the violation has one
of two origins —
  * misread-origin   : the model cannot even formalize that clue correctly in a
                       fresh context (it "read" the puzzle wrong);
  * reasoning-origin : it formalizes the clue correctly, so the error crept in
                       during search/deduction.
Prediction: location feedback ("you broke clue 3") repairs reasoning-origin
errors but not misread-origin ones — feedback that contradicts the model's
internal (wrong) reading gets rationalized away.

Design notes:
- Probes run OFF-PATH (fresh contexts, never shown to the repair branches), so
  classification cannot contaminate the repair measurement.
- Repair arms branch from the SAME failed first attempt:
    binary   : "your grid is wrong" — oracle verdict, no location
    location : names + quotes the violated clue(s) VERBATIM — location only
    interp   : referee-style paraphrase of the violated constraint(s) — hands
               over the correct READING (the strongest feedback)
  If misread-origin errors resist even `interp`, the strong version of the
  hypothesis holds. (The referee pilot used interp-style feedback only.)
- The reading probe is a black-box proxy: restating a clue correctly does not
  prove the model USED that reading while solving. Best available without
  logit access; stated limitation.
- Everything raw is saved: any number can be audited after the fact.
"""

from __future__ import annotations

import json
import re
from math import comb

from zebralogic.referee import cat_alias
from zebralogic.referee import feedback as interp_feedback
from zebralogic.referee import violations
from zebralogic.zebra_experiment import (
    _canon,
    _norm_constraint,
    _parse_grid,
    prompt_A,
    score_answer,
)
from zebralogic.zebra_parser import build, clue_lines, parse_clue, parse_header

MAX_READ_PROBES = 3  # cap reading probes per failure (cost bound; cap noted in row)


# --------------------------------------------------------------------------- #
# Clue <-> constraint mapping
# --------------------------------------------------------------------------- #


def clue_map(nl: str):
    """1-based clue number -> (clue text, [Constraint]); and Constraint -> clue
    number. load_zebra only keeps fully-parsed puzzles, so every solver
    constraint traces back to exactly one numbered clue line."""
    _, cats = parse_header(nl)
    by_no: dict[int, tuple] = {}
    by_con: dict[object, int] = {}
    for no, text in enumerate(clue_lines(nl), start=1):
        cons = parse_clue(text, cats) or []
        by_no[no] = (text, cons)
        for c in cons:
            by_con.setdefault(c, no)
    return by_no, by_con


# --------------------------------------------------------------------------- #
# Probes (off-path: fresh contexts, never fed into the repair branches)
# --------------------------------------------------------------------------- #


def prompt_read(inst: dict, clue_text: str) -> str:
    return (
        f"{inst['nl']}\n\nConsider ONLY this clue from the puzzle above:\n"
        f'"{clue_text}"\n\nExpress it as a formal constraint string '
        '"relation|category::value|category::value|distance" (relations: eq, neq, '
        "left, right, adj, dleft, dist; House::n for a house number; distance only "
        "for dist). Output ONLY a JSON list containing that one string."
    )


_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4",
             "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9"}


def _norm_house_ops(con: tuple) -> tuple:
    """A misread verdict must mean the model misread the CLUE, not the output
    format: fold 'House::first' / 'House::house 2' to digits, and re-tag a
    bare-digit operand as a house reference ('2' for 'House::2')."""
    rel, a, b, d = con

    def fix(op):
        tag, v = op
        m = re.search(r"\d+", v)
        digit = m.group(0) if m else _ORDINALS.get(v.strip())
        if tag == "H":
            return ("H", digit) if digit else op
        return ("H", digit) if (digit == v.strip() and digit) else op

    a, b = fix(a), fix(b)
    if rel in ("eq", "neq", "adj", "dist"):
        a, b = tuple(sorted((a, b)))
    return (rel, a, b, d)


def grade_read(text: str, gold_cons: list) -> dict:
    """Meaning-level match (same normalizer as condition C) of the model's
    formalization against the parser's constraints for that clue."""
    gold = {_norm_constraint(_canon(c)) for c in gold_cons} - {None}
    got: set = set()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            items = arr if isinstance(arr, list) else [arr]
            got = {_norm_constraint(x) for x in items if isinstance(x, str)} - {None}
        except json.JSONDecodeError:
            pass
    if not got and "|" in text:  # bare string without JSON brackets
        got = {_norm_constraint(text.strip().strip('"'))} - {None}
    ok = bool(gold) and (got == gold
                         or {_norm_house_ops(c) for c in got}
                         == {_norm_house_ops(c) for c in gold})
    return {"ok": ok, "got": sorted(map(str, got))}


def prompt_localize(inst: dict, grid: dict) -> str:
    rows = [grid.get(h, {}) for h in range(1, inst["n"] + 1)]
    return (
        f"{inst['nl']}\n\nHere is a CANDIDATE solution grid (house 1 to "
        f"{inst['n']} in order):\n{json.dumps(rows)}\n\nWhich numbered clue(s) "
        "does this grid violate? Check every clue. Output ONLY a JSON list of "
        "the violated clue numbers (e.g. [3, 7])."
    )


def parse_localize(text: str):
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for x in arr:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            return None
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Repair arms
# --------------------------------------------------------------------------- #

FB_BINARY = (
    "Your grid is wrong: it does not satisfy all the clues. Re-check every clue "
    "and fix the grid. Output ONLY the JSON grid."
)


def fb_location(bad: list, by_con: dict, by_no: dict) -> str:
    nos = sorted({by_con[c] for c in bad if c in by_con})
    lines = [f'- Clue {no}: "{by_no[no][0]}"' for no in nos]
    return (
        "Your grid violates these clue(s) from the puzzle:\n"
        + "\n".join(lines)
        + "\nFix the grid so EVERY clue holds. Output ONLY the JSON grid."
    )


def _gen(model, prompt: str, label: str, seed=None) -> str:
    """generate() with graceful degradation for mocks lacking label/seed kwargs."""
    try:
        return model.generate(prompt, label=label, seed=seed)
    except TypeError:
        try:
            return model.generate(prompt, label=label)
        except TypeError:
            return model.generate(prompt)


def run_arm(inst: dict, model, arm: str, first_text: str, constraints, by_con, by_no,
            max_retries: int, alias=None, vary_seed: bool = False,
            base_seed: int | None = None) -> dict:
    """Branch from the same failed first attempt; feedback is recomputed from
    the CURRENT grid's violations each round (binary stays constant).
    vary_seed: new sampling seed per retry — separates 'cannot fix' from
    'fixed seed deterministically reproduced the same grid'."""
    base = prompt_A(inst)
    texts, prev = [], first_text
    bad = violations(constraints, _parse_grid(prev) or {}, alias)
    attempts = 0
    for attempt in range(max_retries):
        if arm == "binary":
            fb = FB_BINARY
        elif arm == "location":
            fb = fb_location(bad, by_con, by_no)
        else:  # interp: referee-style paraphrase of the violated constraints
            fb = interp_feedback(bad)
        prompt = f"{base}\n\nYour previous answer:\n{prev}\n\n{fb}"
        seed = (base_seed or 0) + attempt + 1 if vary_seed else None
        text = _gen(model, prompt, f"origin {inst['id']} {arm} try{attempt + 1}", seed)
        texts.append(text)
        attempts = attempt + 1
        prev = text
        grid = _parse_grid(text)
        bad = list(constraints) if grid is None else violations(constraints, grid, alias)
        if not bad:
            break
    return {
        "texts": texts,
        "attempts": attempts,
        "converged": not bad,
        "final_full": score_answer(prev, inst)["full"] if texts else False,
    }


def run_self_arm(inst: dict, model, first_text: str, constraints, by_no, alias,
                 max_retries: int) -> dict:
    """Self-referee: NO oracle inside the loop. The model checks its own grid
    (localization prompt), names the violated clues itself, and retries on its
    OWN feedback. Stops when it declares its grid clean or retries run out.
    The oracle scores only the FINAL grid, after the loop."""
    base = prompt_A(inst)
    prev, log = first_text, []
    for rnd in range(max_retries):
        grid = _parse_grid(prev)
        raw = _gen(model, prompt_localize(inst, grid or {}),
                   f"closer {inst['id']} selfcheck{rnd + 1}")
        pred = parse_localize(raw)
        if pred == []:  # model believes the grid is clean: self-stop
            log.append({"round": rnd + 1, "self_check": pred, "stop": "self_clean"})
            break
        valid = [n for n in (pred or []) if n in by_no]
        if valid:
            lines = [f'- Clue {n}: "{by_no[n][0]}"' for n in valid]
            fb = ("You re-checked your grid and found it violates these clue(s):\n"
                  + "\n".join(lines)
                  + "\nFix the grid so EVERY clue holds. Output ONLY the JSON grid.")
        else:  # unusable self-check: fall back to a bare retry signal
            fb = FB_BINARY
        text = _gen(model, f"{base}\n\nYour previous answer:\n{prev}\n\n{fb}",
                    f"closer {inst['id']} selffix{rnd + 1}")
        log.append({"round": rnd + 1, "self_check": pred, "raw": text})
        prev = text
    grid = _parse_grid(prev)
    bad = list(constraints) if grid is None else violations(constraints, grid, alias)
    return {
        "log": log,
        "final": prev,
        "rounds": len(log),
        "self_clean_stop": bool(log) and log[-1].get("stop") == "self_clean",
        "converged": not bad,
        "final_full": score_answer(prev, inst)["full"],
    }


# --------------------------------------------------------------------------- #
# Per-instance flow
# --------------------------------------------------------------------------- #


def run_one(inst: dict, model, arms: list[str], max_retries: int = 3) -> dict:
    def gen(prompt, label):
        try:
            return model.generate(prompt, label=label)
        except TypeError:
            return model.generate(prompt)

    constraints = build(inst["nl"])[0].constraints
    by_no, by_con = clue_map(inst["nl"])
    alias = cat_alias(inst)

    first = gen(prompt_A(inst), f"origin {inst['id']} solve")
    s0 = score_answer(first, inst)
    row = {"id": inst["id"], "size": inst["size"], "raw0": first,
           "first_full": s0["full"], "first_cell": s0["cell"]}
    grid0 = _parse_grid(first)
    if s0["full"]:
        row["status"] = "correct"
        return row
    if grid0 is None:
        row["status"] = "no_parse"
        return row

    bad0 = violations(constraints, grid0, alias)
    if not bad0:  # wrong vs gold yet violates nothing: scorer/verifier mismatch
        row["status"] = "anomaly"
        return row
    row["status"] = "failure"
    nos = sorted({by_con[c] for c in bad0 if c in by_con})
    row["violated_clues"] = nos
    row["unmapped_violations"] = sum(1 for c in bad0 if c not in by_con)

    # -- reading probe per violated clue (fresh context each) --
    reads = []
    for no in nos[:MAX_READ_PROBES]:
        text, cons = by_no[no]
        raw = gen(prompt_read(inst, text), f"origin {inst['id']} read{no}")
        g = grade_read(raw, cons)
        reads.append({"clue_no": no, "clue": text, "raw": raw, **g})
    row["reads"] = reads
    row["reads_capped"] = len(nos) > MAX_READ_PROBES
    row["misread"] = any(not r["ok"] for r in reads)

    # -- self-localization probe (fresh context) --
    raw = gen(prompt_localize(inst, grid0), f"origin {inst['id']} localize")
    pred = parse_localize(raw)
    row["localize"] = {"raw": raw, "predicted": pred, "oracle": nos,
                       "hit": bool(pred and set(pred) & set(nos)),
                       "exact": pred == nos}

    # -- repair arms, each branching from the same first attempt --
    row["arms"] = {a: run_arm(inst, model, a, first, constraints, by_con, by_no,
                              max_retries, alias) for a in arms}
    return row


def run_origin(instances: list[dict], model, arms: list[str], max_retries: int = 3,
               workers: int = 1, progress=None, save=None) -> list[dict]:
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows, done, lock = [], 0, threading.Lock()

    def one(inst):
        t0 = time.time()
        return run_one(inst, model, arms, max_retries), time.time() - t0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, i) for i in instances]):
            row, secs = fut.result()
            with lock:
                rows.append(row)
                done += 1
                if progress:
                    slim = {k: v for k, v in row.items()
                            if k not in ("raw0", "reads", "localize", "arms")}
                    progress.item_done("origin", slim, secs)
                if save:
                    save(rows)
            print(f"\r    origin {done}/{len(instances)} puzzles", end="", flush=True)
    sys.stdout.write("\n")
    return rows


# --------------------------------------------------------------------------- #
# Closer: self-referee vs oracle feedback, paired on the SAME failures
# --------------------------------------------------------------------------- #


def run_closer(fail_rows: list[dict], insts: dict, model, max_retries: int = 3,
               workers: int = 1, progress=None, save=None) -> list[dict]:
    """fail_rows: regraded failure rows (status2 == 'failure') from the origin
    run. Each closer row branches from the SAME saved first attempt (raw0):
      self      : self-referee loop, fixed seed   (head-to-head vs stored 'location')
      loc_vseed : oracle location feedback, fresh seed per retry (lock-in control)
    Stored arm outcomes are copied over so analysis is fully paired."""
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base_seed = getattr(model, "seed", 0)
    rows, done, lock = [], 0, threading.Lock()

    def one(r):
        t0 = time.time()
        inst = insts[r["id"]]
        constraints = build(inst["nl"])[0].constraints
        by_no, by_con = clue_map(inst["nl"])
        alias = cat_alias(inst)
        first = r["raw0"]
        out = {
            "id": r["id"], "size": r["size"], "tainted": r.get("tainted", False),
            "misread": r.get("misread2", r.get("misread")),
            "stored": {a: v.get("converged2", v.get("converged"))
                       for a, v in (r.get("arms") or {}).items()},
            "self": run_self_arm(inst, model, first, constraints, by_no, alias,
                                 max_retries),
            "loc_vseed": run_arm(inst, model, "location", first, constraints,
                                 by_con, by_no, max_retries, alias,
                                 vary_seed=True, base_seed=base_seed),
        }
        # genuine insistence: identical grid re-emitted DESPITE a fresh seed
        seq = [_parse_grid(t) for t in [first] + out["loc_vseed"]["texts"]]
        out["loc_vseed"]["insisted"] = any(
            x is not None and x == y for x, y in zip(seq, seq[1:]))
        return out, time.time() - t0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in fail_rows]):
            row, secs = fut.result()
            with lock:
                rows.append(row)
                done += 1
                if progress:
                    slim = {k: row[k] for k in ("id", "size", "tainted", "misread")}
                    slim.update(self_conv=row["self"]["converged"],
                                vseed_conv=row["loc_vseed"]["converged"])
                    progress.item_done("closer", slim, secs)
                if save:
                    save(rows)
            print(f"\r    closer {done}/{len(fail_rows)} failures", end="", flush=True)
    sys.stdout.write("\n")
    return rows


def sign_test_two_sided(b: int, c: int) -> float:
    """Exact two-sided sign test for paired discordant counts b vs c."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = sum(comb(n, k) for k in range(lo + 1)) / 2 ** n * 2
    return min(p, 1.0)


def summarize_closer(rows: list[dict]) -> str:
    out = ["\n=========  CLOSER (self-referee vs oracle)  ========="]
    clean = [r for r in rows if not r["tainted"]]
    out.append(f"  failures rerun: {len(rows)} (clean {len(clean)})")
    for name in ("self", "loc_vseed"):
        k = sum(r[name]["converged"] for r in rows)
        kc = sum(r[name]["converged"] for r in clean)
        extra = ""
        if name == "self":
            fc = sum(1 for r in rows
                     if r["self"]["self_clean_stop"] and not r["self"]["converged"])
            extra = f" | FALSE self-clean stops: {fc}"
        else:
            ins = sum(1 for r in rows
                      if not r[name]["converged"] and r[name]["insisted"])
            tot = sum(1 for r in rows if not r[name]["converged"])
            extra = f" | insisted (same grid, new seed): {ins}/{tot} unconverged"
        out.append(f"  {name:9}: repaired {k}/{len(rows)} (clean {kc}/{len(clean)}){extra}")
    # paired comparisons on clean failures with a stored location outcome
    for name, label in (("self", "self vs oracle-location"),
                        ("loc_vseed", "varied-seed vs fixed-seed location")):
        pairs = [(r[name]["converged"], r["stored"].get("location"))
                 for r in clean if r["stored"].get("location") is not None]
        b = sum(1 for a, s in pairs if a and not s)   # closer arm only
        c = sum(1 for a, s in pairs if s and not a)   # stored arm only
        both = sum(1 for a, s in pairs if a and s)
        p = sign_test_two_sided(b, c)
        out.append(f"  paired {label}: both {both} | only-new {b} | only-stored {c}"
                   f" | sign-test p={p:.3f}")
    mis = [r for r in clean if r["misread"]]
    rea = [r for r in clean if not r["misread"]]
    if mis or rea:
        out.append("  self-arm by origin: misread "
                   f"{sum(r['self']['converged'] for r in mis)}/{len(mis)}"
                   f" | reasoning {sum(r['self']['converged'] for r in rea)}/{len(rea)}")
    if len(clean) < 30:
        out.append(f"  CAUTION: {len(clean)} clean failures — paired test detects only large gaps")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X <= a) for the 2x2 table [[a,b],[c,d]] under fixed margins
    (hypergeometric lower tail). Direction: misread row repairs FEWER."""
    r1, c1, n = a + b, a + c, a + b + c + d
    if n == 0 or comb(n, c1) == 0:
        return 1.0
    lo = max(0, c1 - (n - r1))
    return sum(comb(r1, k) * comb(n - r1, c1 - k) for k in range(lo, a + 1)) / comb(n, c1)


def summarize(rows: list[dict], arms: list[str]) -> str:
    fails = [r for r in rows if r["status"] == "failure"]
    mis = [r for r in fails if r["misread"]]
    rea = [r for r in fails if not r["misread"]]
    out = ["\n=========  ORIGIN (ZebraLogic)  ========="]
    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ("correct", "no_parse", "anomaly", "failure")}
    out.append(f"  instances: {len(rows)} | " + " | ".join(f"{k}: {v}" for k, v in counts.items()))
    out.append(f"  failures by origin: misread {len(mis)} | reasoning {len(rea)}")
    if fails:
        loc = [r["localize"] for r in fails if r.get("localize")]
        hit = sum(x["hit"] for x in loc)
        exact = sum(x["exact"] for x in loc)
        out.append(f"  self-localization: hit {hit}/{len(loc)} | exact {exact}/{len(loc)}")
    for arm in arms:
        def rate(group):
            n = len(group)
            k = sum(1 for r in group if r["arms"][arm]["converged"])
            return k, n
        km, nm = rate(mis)
        kr, nr = rate(rea)
        p = fisher_one_sided(km, nm - km, kr, nr - kr)
        out.append(f"  arm {arm:9}: repaired misread {km}/{nm} | reasoning {kr}/{nr}"
                   f" | one-sided Fisher p={p:.3f}")
        anom = sum(1 for r in fails
                   if r["arms"][arm]["converged"] != r["arms"][arm]["final_full"])
        if anom:
            out.append(f"      WARNING {arm}: {anom} converged/score mismatches — audit raws")
    if len(fails) < 30:
        out.append(f"  CAUTION: only {len(fails)} failures — detects only large effects")
    return "\n".join(out)
