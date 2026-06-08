"""Tests for the sound CSP core (zebralogic.csp).

Three layers, by what they catch:
1. ``ok()`` truth tables  -> the semantics of each constraint kind.
2. Hand-verified puzzles   -> the engine produces *correct* solutions (incl. the
   cross-category ``eq`` link and a real 3x3, not just a 2x2 toy).
3. Property test vs an independent brute-force enumerator -> the
   propagation/search machinery agrees with naive enumeration over hundreds of
   random puzzles.
Plus: input-validation (fail-loud) and the independent ``_verify`` guard.
"""

import random

import pytest

from zebralogic.csp import Constraint, Puzzle, SolverError

# --------------------------------------------------------------------------- #
# 1. ok() truth tables: pin the semantics of every kind directly.
# --------------------------------------------------------------------------- #


def _c(kind: str, d: int = 0) -> Constraint:
    return Constraint(("X", "x"), ("Y", "y"), kind, d)


def test_ok_eq():
    assert _c("eq").ok(2, 2)
    assert not _c("eq").ok(2, 3)


def test_ok_neq():
    assert _c("neq").ok(1, 2)
    assert not _c("neq").ok(2, 2)


def test_ok_left():
    assert _c("left").ok(1, 2)
    assert not _c("left").ok(2, 1)
    assert not _c("left").ok(2, 2)


def test_ok_right():
    assert _c("right").ok(2, 1)
    assert not _c("right").ok(1, 2)
    assert not _c("right").ok(2, 2)


def test_ok_adj():
    assert _c("adj").ok(1, 2) and _c("adj").ok(3, 2)
    assert not _c("adj").ok(1, 3)


def test_ok_notadj():
    assert _c("notadj").ok(1, 3)
    assert not _c("notadj").ok(2, 3)


def test_ok_dleft():
    assert _c("dleft").ok(1, 2)  # b is directly right of a
    assert not _c("dleft").ok(2, 1)
    assert not _c("dleft").ok(1, 3)


def test_ok_dist():
    assert _c("dist", 2).ok(1, 3)  # exactly one house between
    assert not _c("dist", 2).ok(1, 2)
    assert _c("dist", 2).ok(3, 1)


def test_ok_unknown_kind_raises():
    with pytest.raises(ValueError):
        Constraint(("X", "x"), ("Y", "y"), "bogus").ok(1, 2)


# --------------------------------------------------------------------------- #
# 2. Hand-verified puzzles -> the engine solves *correctly*.
# --------------------------------------------------------------------------- #


def _two_by_two() -> Puzzle:
    # Real 2x2: Names Eric/Arnold, Pets dog/cat.
    #   1. Eric is somewhere to the left of Arnold.
    #   2. The person who owns a dog is not in the first house.
    p = Puzzle(2, {"Name": ["Eric", "Arnold"], "Pet": ["dog", "cat"]})
    p.add(Constraint(("Name", "Eric"), ("Name", "Arnold"), "left"))
    p.add(Constraint(("Pet", "dog"), ("House", "1"), "neq"))
    return p


def test_2x2_unique_solution():
    sol = _two_by_two().unique_solution()
    assert sol is not None
    assert sol[("Name", "Eric")] == 1 and sol[("Name", "Arnold")] == 2
    assert sol[("Pet", "cat")] == 1 and sol[("Pet", "dog")] == 2


def test_2x2_grid_render():
    p = _two_by_two()
    grid = p.grid(p.unique_solution())
    assert grid[1] == {"Name": "Eric", "Pet": "cat"}
    assert grid[2] == {"Name": "Arnold", "Pet": "dog"}


def test_cross_category_eq_link():
    # The central, previously-untested mechanic: "X is the <other-attribute>".
    p = Puzzle(2, {"Name": ["A", "B"], "Drink": ["tea", "water"]})
    p.add(Constraint(("Name", "A"), ("House", "1"), "eq"))  # A in house 1
    p.add(Constraint(("Name", "A"), ("Drink", "tea"), "eq"))  # A drinks tea
    grid = p.grid(p.unique_solution())
    assert grid[1] == {"Name": "A", "Drink": "tea"}
    assert grid[2] == {"Name": "B", "Drink": "water"}


def test_3x3_known_solution():
    # Hand-built and hand-verified 3x3 with a unique solution. Exercises
    # absolute position, ordering, and two cross-category eq links at once.
    p = Puzzle(
        3,
        {"Name": ["Arnold", "Eric", "Peter"], "Color": ["red", "blue", "green"]},
    )
    p.add(Constraint(("Name", "Arnold"), ("House", "1"), "eq"))  # Arnold in house 1
    p.add(Constraint(("Name", "Eric"), ("Name", "Peter"), "left"))  # Eric left of Peter
    p.add(Constraint(("Color", "green"), ("House", "1"), "eq"))  # green is house 1
    p.add(Constraint(("Name", "Eric"), ("Color", "red"), "eq"))  # Eric is red
    p.add(Constraint(("Color", "blue"), ("House", "3"), "eq"))  # blue is house 3
    grid = p.grid(p.unique_solution())
    assert grid[1] == {"Name": "Arnold", "Color": "green"}
    assert grid[2] == {"Name": "Eric", "Color": "red"}
    assert grid[3] == {"Name": "Peter", "Color": "blue"}


def test_adjacency_dleft_dist_in_a_real_layout():
    # A 4-house chain pinned by adjacency / directly-left / distance clues.
    p = Puzzle(4, {"Name": ["A", "B", "C", "D"]})
    p.add(Constraint(("Name", "A"), ("House", "1"), "eq"))  # A house 1
    p.add(Constraint(("Name", "A"), ("Name", "B"), "dleft"))  # B directly right of A -> 2
    p.add(Constraint(("Name", "C"), ("Name", "D"), "dleft"))  # D directly right of C
    p.add(Constraint(("Name", "C"), ("House", "3"), "eq"))  # C house 3 -> D house 4
    sol = p.unique_solution()
    assert sol is not None
    assert sol[("Name", "B")] == 2 and sol[("Name", "C")] == 3 and sol[("Name", "D")] == 4


# --------------------------------------------------------------------------- #
# Uniqueness machinery.
# --------------------------------------------------------------------------- #


def test_contradiction_yields_no_solution():
    p = Puzzle(2, {"Name": ["Eric", "Arnold"]})
    p.add(Constraint(("Name", "Eric"), ("Name", "Arnold"), "left"))
    p.add(Constraint(("Name", "Arnold"), ("Name", "Eric"), "left"))
    assert p.solve(limit=2) == []
    assert p.unique_solution() is None


def test_underconstrained_is_not_unique():
    p = Puzzle(2, {"Name": ["Eric", "Arnold"]})
    assert len(p.solve(limit=2)) == 2
    assert p.unique_solution() is None


# --------------------------------------------------------------------------- #
# Input validation: fail loud, never compute on garbage.
# --------------------------------------------------------------------------- #


def test_reject_reserved_house_category():
    with pytest.raises(ValueError):
        Puzzle(2, {"House": ["1", "2"]})


def test_reject_wrong_category_size():
    with pytest.raises(ValueError):
        Puzzle(3, {"Name": ["A", "B"]})  # 2 values, n=3


def test_reject_duplicate_values():
    with pytest.raises(ValueError):
        Puzzle(2, {"Name": ["A", "A"]})


def test_reject_unknown_variable():
    p = Puzzle(2, {"Name": ["A", "B"]})
    with pytest.raises(ValueError):
        p.add(Constraint(("Name", "Z"), ("House", "1"), "eq"))


def test_reject_str_int_house_slip():
    # A common parser bug: passing House 1 as int instead of "1".
    p = Puzzle(2, {"Name": ["A", "B"]})
    with pytest.raises(ValueError):
        p.add(Constraint(("Name", "A"), ("House", 1), "eq"))  # type: ignore[arg-type]


def test_reject_self_loop():
    p = Puzzle(2, {"Name": ["A", "B"]})
    with pytest.raises(ValueError):
        p.add(Constraint(("Name", "A"), ("Name", "A"), "neq"))


def test_reject_unknown_kind_on_add():
    p = Puzzle(2, {"Name": ["A", "B"]})
    with pytest.raises(ValueError):
        p.add(Constraint(("Name", "A"), ("Name", "B"), "sideways"))


# --------------------------------------------------------------------------- #
# The independent _verify guard.
# --------------------------------------------------------------------------- #


def test_verify_accepts_valid_and_rejects_violating():
    p = Puzzle(2, {"Name": ["A", "B"]})
    p.add(Constraint(("Name", "A"), ("Name", "B"), "left"))  # A left of B
    good = {("Name", "A"): 1, ("Name", "B"): 2, ("House", "1"): 1, ("House", "2"): 2}
    assert p._verify(good)
    bad = {("Name", "A"): 2, ("Name", "B"): 1, ("House", "1"): 1, ("House", "2"): 2}
    with pytest.raises(SolverError):
        p._verify(bad)


def test_verify_rejects_non_bijection():
    p = Puzzle(2, {"Name": ["A", "B"]})
    both_house_1 = {
        ("Name", "A"): 1,
        ("Name", "B"): 1,
        ("House", "1"): 1,
        ("House", "2"): 2,
    }
    with pytest.raises(SolverError):
        p._verify(both_house_1)


# --------------------------------------------------------------------------- #
# 3. Property test: engine must agree with independent brute force.
# --------------------------------------------------------------------------- #


def _norm(sols) -> set:
    return {frozenset(s.items()) for s in sols}


def _random_puzzle(rng: random.Random) -> Puzzle:
    n = rng.randint(2, 3)
    ncats = rng.randint(1, 3)
    cats = {f"C{i}": [f"C{i}_{j}" for j in range(n)] for i in range(ncats)}
    p = Puzzle(n, cats)
    allvars = list(p._varset)  # includes House vars
    kinds = ["eq", "neq", "left", "right", "adj", "notadj", "dleft", "dist"]
    for _ in range(rng.randint(0, 5)):
        a, b = rng.choice(allvars), rng.choice(allvars)
        kind = rng.choice(kinds)
        d = rng.randint(1, max(1, n - 1))
        try:
            p.add(Constraint(a, b, kind, d))
        except ValueError:
            pass  # self-loop / dup; just skip
    return p


def test_engine_matches_bruteforce_over_random_puzzles():
    rng = random.Random(20260607)
    for _ in range(500):
        p = _random_puzzle(rng)
        engine = _norm(p.solve(limit=10**6))  # all solutions
        brute = _norm(p.solve_bruteforce())
        assert engine == brute, (
            f"mismatch on constraints={p.constraints} "
            f"engine={len(engine)} brute={len(brute)}"
        )
