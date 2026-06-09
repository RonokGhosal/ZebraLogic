"""TCP experiment harness — the extraction-vs-reasoning decomposition.

No home-built solver needed: TCP ships gold answers AND gold structured fields,
so we measure where the model fails directly.

Three conditions per instance:
  A (baseline)       : model reads the DIALOGUE -> answer.        score vs gold answer
  B (reasoning-only) : model reads the STRUCTURED SPEC -> answer. score vs gold answer
  C (extraction)     : model reads the DIALOGUE -> structured spec. score per field vs gold

The A->B gap tells us how much of the failure was *reading* (extraction); C tells
us *which* facts the model misreads. (Caveat: long-problem structured specs omit
"weekday-only"/exact unavailable dates, so condition B is only clean on short.)

The model is pluggable (`Model` protocol); `MockModel` exercises the pipeline
with zero API/GPU cost. Real runs swap in a vLLM-backed Qwen client.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

# --------------------------------------------------------------------------- #
# Data: normalize each TCP instance to (nl problem, gold spec, gold answer).
# --------------------------------------------------------------------------- #


def load_tcp() -> list[dict]:
    """Return all 600 instances, each annotated with regime/nl/spec/gold."""
    from huggingface_hub import hf_hub_download

    out = []
    for fname, regime in [("TCP_short.jsonl", "short"), ("TCP_long.jsonl", "long")]:
        path = hf_hub_download("Beanbagdzf/TCP", fname, repo_type="dataset")
        for line in open(path):
            if not line.strip():
                continue
            inst = json.loads(line)
            out.append(
                {
                    "id": f"{regime}-{inst['index']}",
                    "regime": regime,
                    "nl": "\n".join(inst["dialogue"]),
                    "question": inst["question"],
                    "spec": gold_spec(inst, regime),
                    "gold": inst["answer"],
                }
            )
    return out


def gold_spec(inst: dict, regime: str) -> dict:
    """The structured facts the dataset provides (the extraction target)."""
    cals = inst["agent_constraints_gmt"] if regime == "short" else (inst.get("agent_constraints") or {})
    start = inst.get("project_start_datetime_gmt") or inst.get("project_start_date")
    return {
        "tasks": dict(inst["tasks"]),
        "dependencies": sorted(tuple(d) for d in inst["dependencies"]),
        "agents": list(inst["agents"]),
        "agent_constraints": cals,
        "project_start": start,
    }


# --------------------------------------------------------------------------- #
# Prompts for the three conditions.
# --------------------------------------------------------------------------- #

_ANS_FMT = (
    "Give ONLY the earliest completion as the last line, formatted exactly "
    "'ANSWER: YYYY-MM-DD HH:MM GMT' (short) or 'ANSWER: YYYY-MM-DD' (long)."
)


def prompt_A(inst: dict) -> str:
    return (
        "A team plans a project in this conversation. Infer the earliest completion "
        "time that satisfies every stated temporal constraint.\n\n"
        f"CONVERSATION:\n{inst['nl']}\n\nQUESTION: {inst['question']}\n\n" + _ANS_FMT
    )


def prompt_B(inst: dict) -> str:
    return (
        "Here are the structured facts of a scheduling problem. Compute the earliest "
        "completion time that satisfies every constraint.\n\n"
        f"FACTS (JSON):\n{json.dumps(inst['spec'], indent=2)}\n\n"
        f"QUESTION: {inst['question']}\n\n" + _ANS_FMT
    )


def prompt_C(inst: dict) -> str:
    return (
        "Read the conversation and extract the scheduling facts as JSON with keys: "
        "tasks (name->duration), dependencies (list of [before, after]), agents (list), "
        "agent_constraints (per agent: working_hours, lunch_break, max_consecutive, "
        "break_after, break_between when stated), project_start.\n\n"
        f"CONVERSATION:\n{inst['nl']}\n\nReturn ONLY the JSON."
    )


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #


def _parse_answer(text: str):
    """Pull a date or datetime out of free model text."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T]+(\d{1,2}):(\d{2})", text)
    if m:
        return ("dt", m.group(1), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return ("date", m.group(1))
    return None


def score_answer(pred_text: str, gold: str, regime: str) -> bool:
    g, p = _parse_answer(gold), _parse_answer(pred_text)
    if not p or not g:
        return False
    if regime == "short":
        return p[0] == "dt" == g[0] and p[1:] == g[1:]
    return p[1] == g[1]


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score_extraction(pred_text: str, spec: dict) -> dict:
    """Per-field correctness of the model's extracted spec vs gold. Each value is
    True/False (or None if unparseable)."""
    pred = _extract_json(pred_text)
    if pred is None:
        return {"parse": False}
    res = {"parse": True}
    res["tasks"] = pred.get("tasks") == spec["tasks"]
    got_deps = sorted(tuple(d) for d in (pred.get("dependencies") or []))
    res["dependencies"] = got_deps == [list(x) for x in spec["dependencies"]] or got_deps == spec["dependencies"]
    # per-agent constraint fields
    gc, pc = spec["agent_constraints"], (pred.get("agent_constraints") or {})
    fields = ["working_hours", "lunch_break", "max_consecutive", "break_after", "break_between"]
    for f in fields:
        gold_vals = {a: gc.get(a, {}).get(f) for a in spec["agents"] if gc.get(a, {}).get(f) is not None}
        if not gold_vals:
            continue
        res[f] = all(
            (pc.get(a, {}) or {}).get(f) == v for a, v in gold_vals.items()
        )
    return res


# --------------------------------------------------------------------------- #
# Model interface (pluggable).
# --------------------------------------------------------------------------- #


class Model(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockModel:
    """Test double. mode='perfect' returns gold; mode='lossy' corrupts extraction."""

    def __init__(self, instances: list[dict], mode: str = "perfect"):
        self.by_prompt = {}
        self.mode = mode
        for inst in instances:
            self.by_prompt[prompt_A(inst)] = "ANSWER: " + inst["gold"]
            self.by_prompt[prompt_B(inst)] = "ANSWER: " + inst["gold"]
            spec = dict(inst["spec"])
            spec["dependencies"] = [list(d) for d in spec["dependencies"]]
            if mode == "lossy":  # drop the first agent's working_hours / break a duration
                spec = json.loads(json.dumps(spec))
                k = next(iter(spec["tasks"]))
                spec["tasks"][k] = spec["tasks"][k] + 99
            self.by_prompt[prompt_C(inst)] = json.dumps(spec)

    def generate(self, prompt: str) -> str:
        return self.by_prompt.get(prompt, "ANSWER: 1900-01-01")

    def generate_many(self, prompts: list[str], label: str = "") -> list[str]:
        return [self.generate(p) for p in prompts]


def run(instances: list[dict], model: Model) -> dict:
    """Run conditions A, B, C over instances; return aggregate stats."""
    stats = {"n": len(instances), "A": 0, "B": 0, "C_field": {}}
    cfield_tot = {}
    for inst in instances:
        if score_answer(model.generate(prompt_A(inst)), inst["gold"], inst["regime"]):
            stats["A"] += 1
        if score_answer(model.generate(prompt_B(inst)), inst["gold"], inst["regime"]):
            stats["B"] += 1
        for f, ok in score_extraction(model.generate(prompt_C(inst)), inst["spec"]).items():
            cfield_tot.setdefault(f, [0, 0])
            cfield_tot[f][1] += 1
            if ok:
                cfield_tot[f][0] += 1
    stats["C_field"] = {f: round(c / n, 3) for f, (c, n) in cfield_tot.items()}
    return stats
