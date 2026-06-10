"""Referee for ZebraLogic: re-check Qwen's grid against the puzzle's own clues,
point at the broken clue, and let Qwen try again — without ever revealing the answer.

Qwen stays the solver. The solver (``csp.Constraint.ok``) is used ONLY to verify
the model's grid and produce targeted feedback, never to produce the answer.

This is sound and non-cheating:
- A ZebraLogic puzzle has exactly one solution, so "satisfies every clue" is a
  complete correctness check (a grid that breaks no clue *is* the unique answer).
- The clues live in the puzzle text, not in a hidden key, so restating which clue
  was violated tells the model nothing it wasn't already given.

The open question this exists to measure: when told "you broke clue X", can the
frozen model actually fix it, or does it loop on the same error?
"""

from __future__ import annotations

from zebralogic.csp import HOUSE
from zebralogic.zebra_experiment import _parse_grid, prompt_A, score_answer
from zebralogic.zebra_parser import build


def _positions(grid: dict) -> dict[str, int]:
    """Qwen's grid {house -> {cat: value}} -> {value(lowercased) -> house}."""
    pos: dict[str, int] = {}
    for h, row in grid.items():
        if isinstance(row, dict):
            for v in row.values():
                pos[str(v).strip().lower()] = h
    return pos


def _pos_of(operand, pos: dict[str, int]):
    cat, val = operand
    if cat == HOUSE:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    return pos.get(str(val).strip().lower())


def violations(constraints, grid: dict) -> list:
    """Constraints the grid breaks (or whose values it failed to place)."""
    pos = _positions(grid)
    bad = []
    for con in constraints:
        pa, pb = _pos_of(con.a, pos), _pos_of(con.b, pos)
        if pa is None or pb is None or not con.ok(pa, pb):
            bad.append(con)
    return bad


_PHRASE = {
    "eq": "{a} must be in the same house as {b}",
    "neq": "{a} must not be in the same house as {b}",
    "left": "{a} must be somewhere to the left of {b}",
    "right": "{a} must be somewhere to the right of {b}",
    "adj": "{a} must be directly next to {b}",
    "notadj": "{a} must not be next to {b}",
    "dleft": "{a} must be immediately to the left of {b}",
    "dist": "{a} must be exactly {d} house(s) away from {b}",
}


def _label(operand) -> str:
    cat, val = operand
    return f"house {val}" if cat == HOUSE else f"'{val}'"


def feedback(bad: list) -> str:
    lines = [
        "- " + _PHRASE.get(c.kind, "{a} ~ {b}").format(a=_label(c.a), b=_label(c.b), d=c.d)
        for c in bad
    ]
    return (
        "Your grid breaks these clues from the puzzle:\n"
        + "\n".join(lines)
        + "\nFix the grid so EVERY clue holds. Output ONLY the JSON grid."
    )


def solve_with_referee(inst: dict, model, max_retries: int = 3) -> dict:
    """Qwen solves; the referee checks its grid against the clues and, on a
    violation, names the broken clue and asks again. Returns all attempt texts +
    metadata. The answer key is never used or revealed."""
    constraints = build(inst["nl"])[0].constraints
    base = prompt_A(inst)
    prompt = base
    texts: list[str] = []
    for attempt in range(max_retries + 1):
        text = model.generate(prompt)
        texts.append(text)
        grid = _parse_grid(text)
        bad = list(constraints) if grid is None else violations(constraints, grid)
        if not bad:
            return {"texts": texts, "final": text, "attempts": attempt + 1, "converged": True}
        prompt = f"{base}\n\nYour previous answer:\n{text}\n\n{feedback(bad)}"
    return {"texts": texts, "final": texts[-1], "attempts": max_retries + 1, "converged": False}


def compare(instances: list[dict], model, max_retries: int = 3, workers: int = 4) -> list[dict]:
    """Raw Qwen (first attempt) vs refereed Qwen, scored against gold. The per-
    puzzle referee loops are independent, so they run concurrently."""
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(inst):
        r = solve_with_referee(inst, model, max_retries)
        return {
            "id": inst["id"],
            "size": inst["size"],
            "raw_full": score_answer(r["texts"][0], inst)["full"],
            "ref_full": score_answer(r["final"], inst)["full"],
            "attempts": r["attempts"],
            "converged": r["converged"],
        }

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, i) for i in instances]):
            rows.append(fut.result())
            done += 1
            print(f"\r    referee {done}/{len(instances)} puzzles", end="", flush=True)
    sys.stdout.write("\n")
    return rows
