"""Sound CSP core for ZebraLogic puzzles.

A puzzle places N people in N houses numbered 1..N. Each *category* (Name, Pet,
Color, ...) has exactly N distinct values, one per house. We model the *position*
of every ``(category, value)`` as a variable whose domain is a subset of
``{1..N}``. House numbers are themselves a category ("House") with fixed
singleton domains, so a clue like "X is in house 3" is simply
``Equal(X, ("House", "3"))`` and every clue becomes a constraint between two
variables.

Framing (the "constructor" vocabulary, made concrete): each :class:`Constraint`
is a reusable transformer that prunes values that have become *impossible*;
propagation to a fixpoint + backtracking search find solutions and detect
contradictions.

Trust model (this is a ground-truth oracle, so it must earn trust, not assume
it):
- :meth:`Puzzle._verify` independently re-checks every emitted solution against
  the constraints, so a bug in propagation/search cannot silently emit a wrong
  answer.
- :meth:`Puzzle.solve_bruteforce` is an independent enumerator that shares no
  propagation/search logic; the test suite cross-checks the two over hundreds of
  random puzzles.
- The backtracking search is complete for the binary relations modeled here, so
  ``solve(limit=2)`` distinguishes 0 / 1 / >=2 solutions reliably.

A caveat we do NOT paper over: a *unique* solution is necessary but **not
sufficient** for a *correct* parse. A systematically wrong parser (e.g. swapping
left/right) yields unique-but-wrong solutions. That class of error is caught not
here but at ground-truth generation, via an external cross-check (a second
solver and spot verification against the puzzle text).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

HOUSE = "House"
KINDS = frozenset({"eq", "neq", "left", "right", "adj", "notadj", "dleft", "dist"})
Var = tuple[str, str]  # (category, value)


class SolverError(RuntimeError):
    """Raised when the solver emits an assignment that fails independent checks."""


@dataclass(frozen=True)
class Constraint:
    """A binary constraint over variables ``a`` and ``b``.

    ``ok(pa, pb)`` decides whether positions ``pa`` (of ``a``) and ``pb`` (of
    ``b``) are mutually consistent. Every clue type reduces to one of these.
    """

    a: Var
    b: Var
    kind: str
    d: int = 0  # distance, only used by kind == "dist"

    def ok(self, pa: int, pb: int) -> bool:
        k = self.kind
        if k == "eq":
            return pa == pb
        if k == "neq":
            return pa != pb
        if k == "left":  # a strictly left of b
            return pa < pb
        if k == "right":  # a strictly right of b
            return pa > pb
        if k == "adj":  # a next to b
            return abs(pa - pb) == 1
        if k == "notadj":
            return abs(pa - pb) != 1
        if k == "dleft":  # a directly left of b (b == a + 1)
            return pb - pa == 1
        if k == "dist":  # exactly d houses apart
            return abs(pa - pb) == self.d
        raise ValueError(f"unknown constraint kind: {k!r}")


class Puzzle:
    """A ZebraLogic CSP: N houses, named categories, and clue constraints."""

    def __init__(self, n: int, categories: dict[str, list[str]]):
        if n < 1:
            raise ValueError("n must be >= 1")
        if HOUSE in categories:
            raise ValueError(f"{HOUSE!r} is a reserved category name")
        for cat, vals in categories.items():
            if len(vals) != n:
                raise ValueError(
                    f"category {cat!r} has {len(vals)} values; expected n={n}"
                )
            if len(set(vals)) != len(vals):
                raise ValueError(f"category {cat!r} has duplicate values")
        self.n = n
        self.categories: dict[str, list[str]] = dict(categories)
        self.categories[HOUSE] = [str(i) for i in range(1, n + 1)]
        self.constraints: list[Constraint] = []
        self._vars: list[Var] = [
            (c, v) for c, vs in self.categories.items() for v in vs
        ]
        self._varset: set[Var] = set(self._vars)

    def add(self, c: Constraint) -> None:
        if c.a not in self._varset:
            raise ValueError(f"constraint references unknown variable {c.a!r}")
        if c.b not in self._varset:
            raise ValueError(f"constraint references unknown variable {c.b!r}")
        if c.a == c.b:
            raise ValueError(f"self-referential constraint on {c.a!r}")
        if c.kind not in KINDS:
            raise ValueError(f"unknown constraint kind {c.kind!r}")
        if c.kind == "dist" and c.d < 1:
            raise ValueError("dist constraint requires d >= 1")
        self.constraints.append(c)

    # ---- solving -------------------------------------------------------------

    def _init_domains(self) -> dict[Var, set[int]]:
        dom: dict[Var, set[int]] = {}
        for c, v in self._vars:
            dom[(c, v)] = {int(v)} if c == HOUSE else set(range(1, self.n + 1))
        return dom

    @staticmethod
    def _revise(dom: dict[Var, set[int]], con: Constraint) -> bool:
        """Make ``con`` arc-consistent in both directions; True if anything pruned."""
        da, db = dom[con.a], dom[con.b]
        new_a = {pa for pa in da if any(con.ok(pa, pb) for pb in db)}
        new_b = {pb for pb in db if any(con.ok(pa, pb) for pa in da)}
        changed = False
        if new_a != da:
            dom[con.a] = new_a
            changed = True
        if new_b != db:
            dom[con.b] = new_b
            changed = True
        return changed

    def _alldiff(self, dom: dict[Var, set[int]]) -> bool:
        """Within each category the N values occupy distinct houses (a bijection)."""
        changed = False
        for cat, vals in self.categories.items():
            vars_ = [(cat, v) for v in vals]
            for vr in vars_:  # a placed value removes its house from its siblings
                if len(dom[vr]) == 1:
                    (h,) = tuple(dom[vr])
                    for other in vars_:
                        if other != vr and h in dom[other]:
                            dom[other].discard(h)
                            changed = True
            for h in range(1, self.n + 1):  # a house only one value can take is its
                holders = [vr for vr in vars_ if h in dom[vr]]
                if len(holders) == 1 and len(dom[holders[0]]) > 1:
                    dom[holders[0]] = {h}
                    changed = True
        return changed

    def _propagate(self, dom: dict[Var, set[int]]) -> bool:
        """Prune to a fixpoint. Returns False on contradiction (an empty domain)."""
        changed = True
        while changed:
            changed = False
            for con in self.constraints:
                if self._revise(dom, con):
                    changed = True
            if self._alldiff(dom):
                changed = True
            if any(not s for s in dom.values()):
                return False
        return True

    def _search(self, dom: dict[Var, set[int]], sols: list, limit: int) -> None:
        if len(sols) >= limit:
            return
        unassigned = [(v, d) for v, d in dom.items() if len(d) > 1]
        if not unassigned:
            sols.append({v: next(iter(d)) for v, d in dom.items()})
            return
        var, d = min(unassigned, key=lambda kv: len(kv[1]))  # MRV
        for val in sorted(d):
            nd = {k: set(s) for k, s in dom.items()}
            nd[var] = {val}
            if self._propagate(nd):
                self._search(nd, sols, limit)
                if len(sols) >= limit:
                    return

    def _verify(self, sol: dict[Var, int]) -> bool:
        """Independently re-check ``sol`` against the clues. Raises on any failure.

        This shares nothing with propagation/search, so a bug there cannot
        produce a silently-wrong "solution" that we'd trust as ground truth.
        """
        for con in self.constraints:
            if not con.ok(sol[con.a], sol[con.b]):
                raise SolverError(f"emitted solution violates constraint {con}")
        for cat, vals in self.categories.items():
            houses = sorted(sol[(cat, v)] for v in vals)
            if houses != list(range(1, self.n + 1)):
                raise SolverError(f"category {cat!r} is not a bijection in solution")
        return True

    def solve(self, limit: int = 2) -> list[dict[Var, int]]:
        """Return up to ``limit`` solutions (each maps Var -> house position).

        Every returned solution is independently verified by :meth:`_verify`.
        """
        dom = self._init_domains()
        sols: list[dict[Var, int]] = []
        if self._propagate(dom):
            self._search(dom, sols, limit)
        for sol in sols:
            self._verify(sol)
        return sols

    def solve_bruteforce(self, limit: int | None = None) -> list[dict[Var, int]]:
        """Independent enumerator (no propagation/search) for cross-checking.

        Assigns each non-House category a permutation of 1..N and keeps the
        assignments that satisfy every constraint. Exponential -- for tests on
        small puzzles only.
        """
        cats = [c for c in self.categories if c != HOUSE]
        choices = [list(permutations(range(1, self.n + 1))) for _ in cats]
        base = {(HOUSE, v): int(v) for v in self.categories[HOUSE]}
        sols: list[dict[Var, int]] = []
        for combo in product(*choices):
            sol = dict(base)
            for c, perm in zip(cats, combo):
                for v, h in zip(self.categories[c], perm):
                    sol[(c, v)] = h
            if all(con.ok(sol[con.a], sol[con.b]) for con in self.constraints):
                sols.append(sol)
                if limit is not None and len(sols) >= limit:
                    break
        return sols

    def unique_solution(self) -> dict[Var, int] | None:
        """Return the solution iff exactly one exists, else None (0 or >1)."""
        sols = self.solve(limit=2)
        return sols[0] if len(sols) == 1 else None

    def grid(self, sol: dict[Var, int]) -> dict[int, dict[str, str]]:
        """Render a solution as house -> {category: value} (House excluded)."""
        cats = [c for c in self.categories if c != HOUSE]
        out: dict[int, dict[str, str]] = {}
        for h in range(1, self.n + 1):
            row: dict[str, str] = {}
            for c in cats:
                for v in self.categories[c]:
                    if sol[(c, v)] == h:
                        row[c] = v
            out[h] = row
        return out
