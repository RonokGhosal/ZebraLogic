"""Sound temporal scheduler for the TCP benchmark — the deterministic oracle.

Each TCP instance has 3 tasks (with durations), a precedence DAG, and 2 agents
with calendars. Goal: the earliest project completion. The problem is tiny, so
we exhaustively try every task->agent assignment and every topological ordering
(<=48 combos) and simulate each agent's calendar; the minimum completion over
all of them is provably the earliest.

Two regimes:
  - "short": hour granularity. Calendar = working_hours (may wrap past midnight,
    e.g. [16, 0]), lunch_break, max_consecutive/break_after, break_between. Uses
    ``agent_constraints_gmt`` (GMT-normalized). Answer "YYYY-MM-DD HH:MM GMT".
  - "long": day granularity. Agents work every day except
    ``agent_unavailable_dates``; max_consecutive/break_after/break_between in
    days. Answer "YYYY-MM-DD".

Validated by reproducing all 600 gold answers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import permutations, product

_SHORT_FMT = "%Y-%m-%d %H:%M"
_DATE_FMT = "%Y-%m-%d"


def _in_window(h: int, start: int, end: int) -> bool:
    """Is hour ``h`` inside [start, end), where end==0 means midnight (24) and
    end<=start means the window wraps past midnight."""
    end = 24 if end == 0 else end
    return start <= h < end if start < end else (h >= start or h < end)


def _topo_orders(tasks, deps):
    out = []
    for perm in permutations(tasks):
        pos = {t: i for i, t in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in deps):
            out.append(perm)
    return out


def _work(is_avail, start_min: int, dur: int, max_consec, break_after: int) -> int:
    """Place ``dur`` working units from ``start_min``; return exclusive-end unit.

    Honors max_consecutive worked units, after which ``break_after`` units of
    rest are forced. Natural gaps (unavailable units) reset the consecutive run.
    """
    u, worked, last, consec = start_min, 0, start_min, 0
    cap = start_min + 500_000
    while worked < dur:
        if u > cap:
            raise RuntimeError("agent has no availability")
        if is_avail(u):
            if max_consec and consec >= max_consec:
                u += break_after  # forced rest of break_after units
                consec = 0
                continue
            worked += 1
            last = u
            consec += 1
        else:
            consec = 0
        u += 1
    return last + 1


def _avail_short(cal, base, u):
    h = (base + timedelta(hours=u)).hour
    s, e = cal["working_hours"]
    if not _in_window(h, s, e):
        return False
    lb = cal.get("lunch_break")
    if lb and _in_window(h, lb[0], lb[1]):
        return False
    return True


def _avail_long(base, u, unavailable):
    return (base + timedelta(days=u)).strftime(_DATE_FMT) not in unavailable


def _dateset(v) -> set:
    if isinstance(v, (list, tuple)):
        return set(v)
    if isinstance(v, str):
        return {v}
    return set()


def solve(inst: dict, regime: str) -> str:
    tasks = list(inst["tasks"].keys())
    durs = inst["tasks"]
    deps = [tuple(x) for x in inst["dependencies"]]
    agents = inst["agents"]

    if regime == "short":
        base = datetime.strptime(inst["project_start_datetime_gmt"], _SHORT_FMT)
        cals = inst["agent_constraints_gmt"]
        avail = {a: (lambda u, c=cals[a]: _avail_short(c, base, u)) for a in agents}
    else:
        base = datetime.strptime(inst["project_start_date"], _DATE_FMT)
        cals = inst.get("agent_constraints") or {}
        ud = inst.get("agent_unavailable_dates") or {}
        unavail = {a: _dateset(ud.get(a)) for a in agents}
        avail = {a: (lambda u, s=unavail[a]: _avail_long(base, u, s)) for a in agents}

    cal_of = {a: (cals.get(a) or {}) for a in agents}
    breakbtw = {a: cal_of[a].get("break_between", 0) for a in agents}
    maxcon = {a: cal_of[a].get("max_consecutive") for a in agents}
    breakaf = {a: cal_of[a].get("break_after", 0) for a in agents}

    best = None
    for order in _topo_orders(tasks, deps):
        for combo in product(agents, repeat=len(tasks)):
            assign = dict(zip(tasks, combo))
            finish, agent_end, agent_used = {}, {a: 0 for a in agents}, {a: False for a in agents}
            for t in order:
                a = assign[t]
                prereq = max([finish[p] for (p, s) in deps if s == t] or [0])
                gap = breakbtw[a] if agent_used[a] else 0
                start_min = max(prereq, agent_end[a] + gap, 0)
                end = _work(avail[a], start_min, durs[t], maxcon[a], breakaf[a])
                finish[t], agent_end[a], agent_used[a] = end, end, True
            makespan = max(finish.values())
            if best is None or makespan < best:
                best = makespan

    if regime == "short":
        return (base + timedelta(hours=best)).strftime(_SHORT_FMT) + " GMT"
    return (base + timedelta(days=best - 1)).strftime(_DATE_FMT)
