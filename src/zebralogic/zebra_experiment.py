"""ZebraLogic experiment harness — the SAME A/B/C decomposition as TCP, on the
second (contrasting) benchmark.

Gold solutions + gold constraints come from our parser/solver (the dataset
withholds solutions); only puzzles that parse to a unique solution are included
(774/1000; coverage drops with size, so report stratified by size).

  A (baseline)       : model reads the PUZZLE (clues in prose) -> solution grid.
  B (reasoning-only) : model reads the FORMAL constraints      -> solution grid.
  C (extraction)     : model reads the puzzle -> formal constraints (vs gold).

Mirrors zebra/tcp parity; mock-validated before any GPU run.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from zebralogic.zebra_parser import build, parse_header


def _canon(con) -> str:
    return f"{con.kind}|{con.a[0]}::{con.a[1]}|{con.b[0]}::{con.b[1]}|{con.d}"


def load_zebra(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ZebraLogicBench", "grid_mode")["test"]
    out = []
    for e in ds:
        try:
            P, un = build(e["puzzle"])
            if un:
                continue
            sols = P.solve(limit=2)
            if len(sols) != 1:
                continue
            grid = P.grid(sols[0])  # {house:int -> {long_cat: value}}
        except Exception:
            continue
        header = e["solution"]["header"]  # ['House', <short cat names...>]
        n, cats = parse_header(e["puzzle"])
        longcats = list(cats.keys())
        if len(longcats) != len(header) - 1:
            continue
        l2s = {lc: header[i + 1] for i, lc in enumerate(longcats)}
        out.append(
            {
                "id": e["id"],
                "size": e["size"],
                "n": n,
                "nl": e["puzzle"],
                "categories": {l2s[c]: vs for c, vs in cats.items()},
                "constraints": sorted(_canon(c) for c in P.constraints),
                "gold": {h: {l2s[c]: v for c, v in row.items()} for h, row in grid.items()},
            }
        )
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def prompt_A(inst: dict) -> str:
    cols = ", ".join(inst["categories"])
    return (
        f"{inst['nl']}\n\nSolve the puzzle. Output ONLY a JSON list of {inst['n']} "
        f"objects (house 1 to {inst['n']} in order), each mapping every category "
        f"({cols}) to its value."
    )


def prompt_B(inst: dict) -> str:
    facts = {
        "houses": inst["n"],
        "categories": inst["categories"],
        "constraints": inst["constraints"],
    }
    return (
        "Here is a logic-grid puzzle as formal facts (constraints use pos(category::value) "
        "with relations eq/neq/left/right/adj/dleft/dist).\n\n"
        f"{json.dumps(facts, indent=1)}\n\nOutput ONLY a JSON list of "
        f"{inst['n']} objects (house 1..N) mapping each category to its value."
    )


def prompt_C(inst: dict) -> str:
    return (
        f"{inst['nl']}\n\nExtract EVERY clue as a formal constraint of the form "
        '"relation|category::value|category::value|distance" (relations: eq, neq, '
        "left, right, adj, dleft, dist; House::n for a house number). Output ONLY a "
        "JSON list of those strings."
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _parse_grid(text: str):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return {i: row for i, row in enumerate(arr, start=1) if isinstance(row, dict)}


def score_answer(text: str, inst: dict) -> dict:
    pred = _parse_grid(text)
    gold = inst["gold"]
    total = correct = 0
    for h, row in gold.items():
        for c, v in row.items():
            total += 1
            got = (pred.get(h) if pred else {}) or {}
            if str(got.get(c, "")).strip().lower() == str(v).strip().lower():
                correct += 1
    return {"full": correct == total, "cell": correct / total if total else 0.0}


def _norm_constraint(s: str):
    """Canonicalize a constraint string so extraction is graded by MEANING, not
    exact format: drop the category label (the model names categories its own
    way), lowercase values, fold 'right'->reversed 'left', and sort the operands
    of symmetric relations. (Caveat: dropping the category slightly over-credits
    when one value lives in two categories, e.g. 'red'; rare.)"""
    parts = s.split("|")
    if len(parts) < 3:
        return None
    rel = parts[0].strip().lower()

    def op(x: str):
        x = x.strip()
        cat, val = x.split("::", 1) if "::" in x else ("", x)
        val = val.strip().lower()
        return ("H", val) if cat.strip().lower() == "house" else ("V", val)

    a, b = op(parts[1]), op(parts[2])
    dpart = parts[3].strip() if len(parts) > 3 else ""
    d = int(dpart) if dpart.lstrip("-").isdigit() else 0
    if rel in ("right", "dright"):
        rel, a, b = ("left" if rel == "right" else "dleft"), b, a
    if rel in ("eq", "neq", "adj", "dist"):
        a, b = tuple(sorted((a, b)))
    return (rel, a, b, d)


def score_extraction(text: str, inst: dict) -> dict:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    gold = {_norm_constraint(s) for s in inst["constraints"]} - {None}
    if not m:
        return {"recall": 0.0, "exact": False}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"recall": 0.0, "exact": False}
    got = ({_norm_constraint(x) for x in arr if isinstance(x, str)} - {None}) if isinstance(arr, list) else set()
    return {"recall": len(got & gold) / len(gold) if gold else 1.0, "exact": got == gold}


# --------------------------------------------------------------------------- #
# Model interface + runner
# --------------------------------------------------------------------------- #


class Model(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockModel:
    def __init__(self, instances: list[dict], mode: str = "perfect"):
        self.by_prompt = {}
        for inst in instances:
            rows = [inst["gold"][h] for h in range(1, inst["n"] + 1)]
            cons = list(inst["constraints"])
            if mode == "lossy":
                rows = json.loads(json.dumps(rows))
                first_cat = next(iter(inst["categories"]))
                if len(rows) >= 2:  # swap one value between two houses -> wrong cells
                    rows[0][first_cat], rows[1][first_cat] = rows[1][first_cat], rows[0][first_cat]
                cons = cons[1:]  # drop one constraint
            self.by_prompt[prompt_A(inst)] = json.dumps(rows)
            self.by_prompt[prompt_B(inst)] = json.dumps(rows)
            self.by_prompt[prompt_C(inst)] = json.dumps(cons)

    def generate(self, prompt: str) -> str:
        return self.by_prompt.get(prompt, "[]")

    def generate_many(self, prompts: list[str], label: str = "") -> list[str]:
        return [self.generate(p) for p in prompts]


def run(instances: list[dict], model: Model) -> dict:
    n = len(instances)
    agg = {"n": n, "A_full": 0, "A_cell": 0.0, "B_full": 0, "B_cell": 0.0, "C_recall": 0.0, "C_exact": 0}
    for inst in instances:
        a = score_answer(model.generate(prompt_A(inst)), inst)
        b = score_answer(model.generate(prompt_B(inst)), inst)
        c = score_extraction(model.generate(prompt_C(inst)), inst)
        agg["A_full"] += a["full"]
        agg["A_cell"] += a["cell"]
        agg["B_full"] += b["full"]
        agg["B_cell"] += b["cell"]
        agg["C_recall"] += c["recall"]
        agg["C_exact"] += c["exact"]
    for k in ("A_cell", "B_cell", "C_recall"):
        agg[k] = round(agg[k] / n, 3) if n else 0.0
    return agg
