"""Parse a ZebraLogic puzzle into a csp.Puzzle, and generate its gold solution.

The dataset withholds solutions, so we make them: parse the clues into
constraints, solve, and the *unique* solution is the gold answer. A correct
parse yields exactly one solution — that uniqueness is our self-check.

References in clues are resolved to (category, value) by the value they contain;
when a value belongs to several categories (e.g. "red" is both a favorite color
and a hair color) we disambiguate using the clue's phrasing.
"""

from __future__ import annotations

import re

from zebralogic.csp import HOUSE, Constraint, Puzzle

_ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
_ORDNUM = {o: i for i, o in enumerate(_ORD, start=1)}
_NUMWORD = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_ALIAS = {"swede": "swedish", "dane": "danish"}
# attributes whose clue phrasing always names them explicitly; only chosen when cued
_MARKERS = {"hair", "mother", "mothers", "child", "children", "birthday", "month"}
_STOP = {
    "each", "person", "people", "has", "have", "unique", "the", "different",
    "houses", "house", "are", "something", "everyone", "lives", "style", "name",
    "names", "keeps", "owns", "who", "whose", "favorite",
}


def parse_header(puzzle: str):
    n = int(re.search(r"There are (\d+) houses", puzzle).group(1))
    cats: dict[str, list[str]] = {}
    for line in puzzle.splitlines():
        s = line.strip().lstrip("-").strip()
        m = re.match(r"(.+?):\s*(.+)$", s)
        if m and "`" in m.group(2):
            cats[m.group(1)] = re.findall(r"`([^`]+)`", m.group(2))
    return n, cats


def clue_lines(puzzle: str) -> list[str]:
    out = []
    for line in puzzle.splitlines():
        m = re.match(r"^\s*\d+\.\s*(.+)$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _distinctive(desc: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", desc.lower()) if len(w) > 3 and w not in _STOP]


def resolve(ref: str, cats: dict[str, list[str]]):
    """A reference -> (category, value) or (HOUSE, n), else None."""
    ref = ref.strip().rstrip(".").strip()
    low = ref.lower()
    if "house" in low:
        for o, i in _ORDNUM.items():
            if re.search(rf"\b{o}\b", low):
                return (HOUSE, str(i))
        m = re.search(r"house (\d+)", low)
        if m:
            return (HOUSE, m.group(1))

    valcats: dict[str, list[tuple[str, str]]] = {}
    for d, vs in cats.items():
        for v in vs:
            valcats.setdefault(v.lower(), []).append((d, v))

    def matches(vl: str) -> bool:
        e = re.escape(vl)
        if re.search(rf"\b{e}s?\b", ref, re.IGNORECASE):
            return True
        if len(vl) >= 3 and re.search(rf"\b{e}[a-z]+\b", ref, re.IGNORECASE):
            return True
        a = _ALIAS.get(vl)
        return bool(a and re.search(rf"\b{a}\b", ref, re.IGNORECASE))

    vl = next((v for v in sorted(valcats, key=len, reverse=True) if matches(v)), None)
    if vl is None:
        return None
    cands = valcats[vl]
    if len(cands) == 1:
        return cands[0]

    # colliding value: pick the category whose distinctive words appear in the clue
    scored = [(sum(w in low for w in _distinctive(d)), d, orig) for d, orig in cands]
    best = max(s for s, _, _ in scored)
    if best > 0:
        return next((d, orig) for s, d, orig in scored if s == best)
    # no cue -> prefer a "primary" attribute (no marker word like 'hair')
    for _, d, orig in scored:
        if not any(w in _MARKERS for w in _distinctive(d)):
            return (d, orig)
    return cands[0]


def parse_clue(c: str, cats: dict[str, list[str]]):
    R = lambda x: resolve(x, cats)  # noqa: E731

    def two(a, b, kind, d=0):
        ra, rb = R(a), R(b)
        return [Constraint(ra, rb, kind, d)] if ra and rb and ra != rb else None

    m = re.match(r"^(.+) is (not in|in) the (\w+) house\.?$", c)
    if m:
        ra = R(m.group(1))
        h = _ORDNUM.get(m.group(3))
        kind = "neq" if "not" in m.group(2) else "eq"
        return [Constraint(ra, (HOUSE, str(h)), kind)] if ra and h else None

    for pat, kind, swap in [
        (r"^(.+) is somewhere to the left of (.+)\.?$", "left", False),
        (r"^(.+) is somewhere to the right of (.+)\.?$", "right", False),
        (r"^(.+) is directly left of (.+)\.?$", "dleft", False),
        (r"^(.+) is directly right of (.+)\.?$", "dleft", True),
        (r"^(.+) and (.+) are next to each other\.?$", "adj", False),
    ]:
        m = re.match(pat, c)
        if m:
            a, b = (m.group(2), m.group(1)) if swap else (m.group(1), m.group(2))
            return two(a, b, kind)

    m = re.match(r"^There is one house between (.+) and (.+)\.?$", c)
    if m:
        return two(m.group(1), m.group(2), "dist", 2)
    m = re.match(r"^There are (\w+) houses between (.+) and (.+)\.?$", c)
    if m and m.group(1) in _NUMWORD:
        return two(m.group(2), m.group(3), "dist", _NUMWORD[m.group(1)] + 1)

    for sp in [mm.start() for mm in re.finditer(r" is ", c)]:
        ra, rb = R(c[:sp]), R(c[sp + 4 :])
        if ra and rb and ra != rb:
            return [Constraint(ra, rb, "eq")]
    return None


def build(puzzle: str):
    """Return (Puzzle, list_of_unparsed_clues)."""
    n, cats = parse_header(puzzle)
    p = Puzzle(n, cats)
    unparsed = []
    for cl in clue_lines(puzzle):
        cons = parse_clue(cl, cats)
        if not cons:
            unparsed.append(cl)
            continue
        for con in cons:
            try:
                p.add(con)
            except ValueError:
                unparsed.append(cl)
    return p, unparsed
