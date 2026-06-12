"""Tests for the origin experiment (zebralogic.origin_experiment).

Layers, by what they catch:
1. clue_map / grade_read / parse_localize / feedback text -> the measurement
   instruments themselves (a broken probe would silently misclassify origins).
2. run_one end-to-end with scripted models -> the per-instance flow: correct
   solve, reasoning-origin failure that repairs, misread-origin classification.
3. fisher_one_sided -> the tail probability against hand-computed values.
"""

import json

from zebralogic.origin_experiment import (
    FB_BINARY,
    clue_map,
    fb_location,
    fisher_one_sided,
    grade_read,
    parse_localize,
    prompt_localize,
    prompt_read,
    run_one,
)
from zebralogic.zebra_parser import build

NL = """There are 2 houses, numbered 1 to 2 from left to right.
 - Name: `Eric`, `Arnold`
 - Pet: `dog`, `cat`

1. Eric is somewhere to the left of Arnold.
2. The person who owns a dog is not in the first house.
"""

INST = {
    "id": "t1",
    "size": "2*2",
    "n": 2,
    "nl": NL,
    "categories": {"Name": ["Eric", "Arnold"], "Pet": ["dog", "cat"]},
    "gold": {1: {"Name": "Eric", "Pet": "cat"}, 2: {"Name": "Arnold", "Pet": "dog"}},
}

GOLD_ROWS = json.dumps([INST["gold"][1], INST["gold"][2]])
# swaps the pets: dog lands in house 1, violating clue 2 (and only clue 2)
WRONG_ROWS = json.dumps(
    [{"Name": "Eric", "Pet": "dog"}, {"Name": "Arnold", "Pet": "cat"}]
)


def test_synthetic_puzzle_is_sound():
    p, unparsed = build(NL)
    assert not unparsed
    assert p.unique_solution() is not None


# --------------------------------------------------------------------------- #
# 1. Instruments
# --------------------------------------------------------------------------- #


def test_clue_map_round_trip():
    by_no, by_con = clue_map(NL)
    assert set(by_no) == {1, 2}
    assert "Eric" in by_no[1][0] and "dog" in by_no[2][0]
    for no, (_, cons) in by_no.items():
        for c in cons:
            assert by_con[c] == no


def test_grade_read_accepts_meaning_level_match():
    by_no, _ = clue_map(NL)
    gold2 = by_no[2][1]  # neq(Pet::dog, House::1)
    assert grade_read('["neq|Pet::dog|House::1|0"]', gold2)["ok"]
    # operand order must not matter for symmetric relations
    assert grade_read('["neq|House::1|Pet::dog|0"]', gold2)["ok"]
    # wrong relation = misread
    assert not grade_read('["eq|Pet::dog|House::1|0"]', gold2)["ok"]
    # garbage = misread
    assert not grade_read("no idea", gold2)["ok"]


def test_parse_localize():
    assert parse_localize("[2]") == [2]
    assert parse_localize('The violated clues are: ["2", 1]') == [1, 2]
    assert parse_localize("none of them") is None


def test_feedback_texts():
    by_no, by_con = clue_map(NL)
    p, _ = build(NL)
    grid = {i: row for i, row in enumerate(json.loads(WRONG_ROWS), start=1)}
    from zebralogic.referee import violations

    bad = violations(p.constraints, grid)
    loc = fb_location(bad, by_con, by_no)
    assert "Clue 2" in loc and by_no[2][0] in loc  # quotes the clue verbatim
    assert "Clue 1" not in loc  # clue 1 holds; must not be named
    # binary must not leak which clue broke
    assert "Clue" not in FB_BINARY and "dog" not in FB_BINARY


# --------------------------------------------------------------------------- #
# 2. run_one end-to-end with scripted models
# --------------------------------------------------------------------------- #


class ScriptedModel:
    """Routes by prompt shape: solve -> wrong grid; probes -> scripted
    answers; any repair turn -> gold grid."""

    def __init__(self, read_answer: str):
        self.read_answer = read_answer

    def generate(self, prompt: str, label: str = "") -> str:
        if "Consider ONLY this clue" in prompt:
            return self.read_answer
        if "CANDIDATE solution grid" in prompt:
            return "[2]"
        if "Your previous answer" in prompt:
            return GOLD_ROWS
        return WRONG_ROWS


def test_run_one_correct_short_circuits():
    class Perfect:
        def generate(self, prompt, label=""):
            return GOLD_ROWS

    row = run_one(INST, Perfect(), ["binary"])
    assert row["status"] == "correct" and "arms" not in row


def test_run_one_reasoning_origin_failure_repairs():
    model = ScriptedModel('["neq|Pet::dog|House::1|0"]')  # reads clue 2 fine
    row = run_one(INST, model, ["binary", "location", "interp"])
    assert row["status"] == "failure"
    assert row["violated_clues"] == [2]
    assert row["unmapped_violations"] == 0
    assert row["misread"] is False  # correct restatement -> reasoning-origin
    assert row["localize"]["hit"] and row["localize"]["exact"]
    for arm in ("binary", "location", "interp"):
        a = row["arms"][arm]
        assert a["converged"] and a["final_full"] and a["attempts"] == 1


def test_run_one_misread_origin_classification():
    model = ScriptedModel('["eq|Pet::dog|House::1|0"]')  # flips neq -> eq
    row = run_one(INST, model, ["binary"])
    assert row["status"] == "failure" and row["misread"] is True


def test_run_one_unparseable_first_answer():
    class Garbage:
        def generate(self, prompt, label=""):
            return "I cannot solve this."

    row = run_one(INST, Garbage(), ["binary"])
    assert row["status"] == "no_parse" and "arms" not in row


# --------------------------------------------------------------------------- #
# Cross-category value collisions: the oracle must look up by (category, value)
# first — 21/774 dataset instances repeat a value across categories.
# --------------------------------------------------------------------------- #

NL_COLLIDE = """There are 2 houses, numbered 1 to 2 from left to right.
 - Each person has a favorite color: `red`, `blue`
 - People have unique hair colors: `red`, `brown`

1. The person whose favorite color is red is in the first house.
2. The person who has red hair is not in the first house.
"""


def test_violations_disambiguate_colliding_values():
    from zebralogic.referee import cat_alias, violations

    p, unparsed = build(NL_COLLIDE)
    assert not unparsed
    inst = {"nl": NL_COLLIDE, "categories": {"Color": ["red", "blue"],
                                             "HairColor": ["red", "brown"]}}
    alias = cat_alias(inst)
    # correct grid: favorite-red in house 1, red hair in house 2
    good = {1: {"Color": "red", "HairColor": "brown"},
            2: {"Color": "blue", "HairColor": "red"}}
    assert violations(p.constraints, good, alias) == []
    # value-only legacy lookup gets this WRONG (red maps to one house only)
    assert violations(p.constraints, good) != []
    # and a genuinely wrong grid is still caught with the alias
    bad = {1: {"Color": "blue", "HairColor": "red"},
           2: {"Color": "red", "HairColor": "brown"}}
    assert violations(p.constraints, bad, alias) != []


def test_grade_read_format_tolerance():
    by_no, _ = clue_map(NL)
    gold2 = by_no[2][1]  # neq(Pet::dog, House::1)
    # ordinal house word, 'house 1' phrasing, and a bare digit must not
    # classify a correct reading as a misread
    assert grade_read('["neq|Pet::dog|House::first|0"]', gold2)["ok"]
    assert grade_read('["neq|Pet::dog|House::house 1|0"]', gold2)["ok"]
    assert grade_read('["neq|Pet::dog|1|0"]', gold2)["ok"]
    # but a genuinely different house number stays a misread
    assert not grade_read('["neq|Pet::dog|House::second|0"]', gold2)["ok"]


# --------------------------------------------------------------------------- #
# 3. Fisher tail
# --------------------------------------------------------------------------- #


def test_fisher_one_sided():
    # all margins balanced, no effect -> p = 1 (lower tail includes everything)
    assert fisher_one_sided(5, 0, 0, 5) == 1.0
    # 0/5 repaired vs 5/5 repaired: p = 1/C(10,5)
    assert abs(fisher_one_sided(0, 5, 5, 0) - 1 / 252) < 1e-12
    # empty table degrades to 1, not a crash
    assert fisher_one_sided(0, 0, 0, 0) == 1.0
