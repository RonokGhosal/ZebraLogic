"""Live run dashboard for monitoring agents.

Writes three artifacts, updated after EVERY model call and EVERY scored item,
atomically (tmp + rename), so an external watcher can poll them at any moment:

  PROGRESS.md           - human/agent-readable dashboard with explicit ALERTS
  progress/status.json  - the same state, machine-readable
  progress/calls.jsonl  - one line per completed model call (append-only)

Staleness rule for watchers: if status is "running" and updated_at is older
than ~30 minutes, the run is stalled (check the run process and the Ollama
server) — single calls on this hardware legitimately take up to ~25 min.
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else 0.0


class Progress:
    def __init__(self, config: dict | None = None, root: str = "."):
        self.root = root
        self.dir = os.path.join(root, "progress")
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.dir, exist_ok=True)
        self.lock = threading.Lock()
        self.t0 = time.time()
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, cwd=root, timeout=5).stdout.strip()
        except Exception:
            rev = "?"
        self.state: dict = {
            "status": "running",
            "phase": "starting",
            "pid": os.getpid(),
            "git_rev": rev,
            "config": config or {},
            "started_at": _now(),
            "updated_at": _now(),
            "calls": {"done": 0, "errors": 0, "truncated": 0, "output_tokens": 0,
                      "avg_seconds": 0.0, "max_seconds": 0.0, "avg_tok_s": 0.0},
            "items": {},      # bench -> {"done": n, "total": n, "avg_seconds": s, "eta_minutes": m}
            "results": {},    # bench -> aggregate scores (computed in render)
            "alerts": [],
            "recent_calls": [],   # last 10
            "recent_errors": [],  # last 5
        }
        self._call_secs: list[float] = []
        self._tok_s: list[float] = []
        self._item_secs: dict[str, list[float]] = collections.defaultdict(list)
        self._rows: dict[str, list[dict]] = collections.defaultdict(list)
        self._render()

    # ------------------------------------------------------------------ hooks

    def set_phase(self, phase: str):
        with self.lock:
            self.state["phase"] = phase
            self._render()

    def add_total(self, bench: str, total: int):
        with self.lock:
            self.state["items"][bench] = {"done": 0, "total": total, "avg_seconds": 0.0,
                                          "eta_minutes": None}
            self._render()

    def call_done(self, label: str, meta: dict, seconds: float):
        with self.lock:
            c = self.state["calls"]
            c["done"] += 1
            c["output_tokens"] += meta.get("output_tokens", 0)
            if meta.get("truncated"):
                c["truncated"] += 1
            if meta.get("error"):
                c["errors"] += 1
                self.state["recent_errors"] = (self.state["recent_errors"]
                                               + [{"at": _now(), "label": label,
                                                   "error": str(meta["error"])[:300]}])[-5:]
            self._call_secs.append(seconds)
            if seconds > 0 and meta.get("output_tokens"):
                self._tok_s.append(meta["output_tokens"] / seconds)
            c["avg_seconds"] = round(_mean(self._call_secs), 1)
            c["max_seconds"] = round(max(self._call_secs), 1)
            c["avg_tok_s"] = round(_mean(self._tok_s), 1)
            self.state["recent_calls"] = (self.state["recent_calls"]
                                          + [{"at": _now(), "label": label,
                                              "seconds": round(seconds, 1),
                                              "output_tokens": meta.get("output_tokens", 0),
                                              "truncated": bool(meta.get("truncated"))}])[-10:]
            with open(os.path.join(self.dir, "calls.jsonl"), "a") as f:
                f.write(json.dumps({"at": _now(), "label": label, "seconds": round(seconds, 1),
                                    **{k: meta.get(k) for k in
                                       ("prompt_tokens", "output_tokens", "truncated", "error")}}) + "\n")
            self._render()

    def item_done(self, bench: str, row: dict, seconds: float):
        """row should already exclude bulky raw outputs."""
        with self.lock:
            self._rows[bench].append(row)
            self._item_secs[bench].append(seconds)
            it = self.state["items"].setdefault(
                bench, {"done": 0, "total": len(self._rows[bench]), "avg_seconds": 0.0,
                        "eta_minutes": None})
            it["done"] = len(self._rows[bench])
            it["avg_seconds"] = round(_mean(self._item_secs[bench]), 1)
            remaining = max(it["total"] - it["done"], 0)
            it["eta_minutes"] = round(remaining * it["avg_seconds"] / 60, 1)
            self._render()

    def finish(self, ok: bool = True, note: str = ""):
        with self.lock:
            self.state["status"] = "done" if ok else "failed"
            self.state["phase"] = note or self.state["phase"]
            self._render()

    # ---------------------------------------------------------------- render

    def _aggregates(self) -> dict:
        out = {}
        zr = self._rows.get("zebra", [])
        if zr:
            by = collections.defaultdict(list)
            for r in zr:
                by[r["size"]].append(r)
            out["zebra"] = {
                "n": len(zr),
                "A_full": _mean(r["A_full"] for r in zr), "A_cell": _mean(r["A_cell"] for r in zr),
                "B_full": _mean(r["B_full"] for r in zr), "B_cell": _mean(r["B_cell"] for r in zr),
                "C_recall": _mean(r["C_recall"] for r in zr),
                "truncated_rows": sum(1 for r in zr if r.get("A_trunc") or r.get("B_trunc") or r.get("C_trunc")),
                "by_size": {s: {"n": len(g),
                                "A_full": _mean(r["A_full"] for r in g),
                                "B_full": _mean(r["B_full"] for r in g),
                                "C_recall": _mean(r["C_recall"] for r in g)}
                            for s, g in sorted(by.items())},
            }
        tr = self._rows.get("tcp", [])
        if tr:
            cf = collections.defaultdict(list)
            for r in tr:
                for f, ok in r["C"].items():
                    cf[f].append(bool(ok))
            out["tcp"] = {"n": len(tr), "A": _mean(r["A"] for r in tr), "B": _mean(r["B"] for r in tr),
                          "C_fields": {f: _mean(v) for f, v in cf.items()}}
        rr = self._rows.get("referee", [])
        if rr:
            by = collections.defaultdict(list)
            for r in rr:
                by[r["size"]].append(r)
            out["referee"] = {
                "n": len(rr),
                "raw_full": _mean(r["raw_full"] for r in rr),
                "ref_full": _mean(r["ref_full"] for r in rr),
                "lift": round(_mean(r["ref_full"] for r in rr) - _mean(r["raw_full"] for r in rr), 3),
                "converged": _mean(r["converged"] for r in rr),
                "avg_attempts": _mean(r["attempts"] for r in rr),
                "by_size": {s: {"n": len(g),
                                "raw_full": _mean(r["raw_full"] for r in g),
                                "ref_full": _mean(r["ref_full"] for r in g)}
                            for s, g in sorted(by.items())},
            }
        return out

    def _alerts(self) -> list[str]:
        a = []
        c = self.state["calls"]
        if c["truncated"]:
            a.append(f"TRUNCATION: {c['truncated']} call(s) hit the num_predict ceiling - those "
                     "items score wrong by construction; raise --max-tokens before trusting results")
        if c["errors"]:
            a.append(f"ERRORS: {c['errors']} failed call(s) - see recent_errors; check the Ollama server")
        if c["max_seconds"] > 2100:
            a.append(f"SLOW CALL: slowest call took {round(c['max_seconds']/60)} min - "
                     "watch for the 60-min client timeout")
        zr = self._rows.get("zebra", [])
        empt = sum(1 for r in zr if not r.get("A_tokens") and not r.get("A_trunc"))
        if zr and empt > len(zr) * 0.2:
            a.append(f"SUSPICIOUS: {empt}/{len(zr)} zebra rows have 0 output tokens (mock run, or "
                     "responses not reaching the scorer)")
        return a

    def _render(self):
        s = self.state
        s["updated_at"] = _now()
        s["elapsed_minutes"] = round((time.time() - self.t0) / 60, 1)
        s["results"] = self._aggregates()
        s["alerts"] = self._alerts()
        self._write(os.path.join(self.dir, "status.json"), json.dumps(s, indent=1))
        self._write(os.path.join(self.root, "PROGRESS.md"), self._markdown())

    @staticmethod
    def _write(path: str, content: str):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, path)

    def _markdown(self) -> str:
        s = self.state
        L = [
            "# Run Progress Dashboard",
            "",
            f"_Auto-written by the harness after every model call. Machine-readable: "
            f"`progress/status.json`; per-call log: `progress/calls.jsonl`._",
            "",
            f"**Watcher rule:** status `running` + `updated_at` older than ~30 min = stalled "
            f"(check `ps {s['pid']}` and `curl localhost:11434/api/version`). Single calls can "
            f"legitimately take ~25 min on this hardware.",
            "",
            f"- **Status:** {s['status']}  |  **Phase:** {s['phase']}",
            f"- **Started:** {s['started_at']}  |  **Updated:** {s['updated_at']}  |  "
            f"**Elapsed:** {s['elapsed_minutes']} min",
            f"- **PID:** {s['pid']}  |  **Git:** {s['git_rev']}",
            f"- **Config:** `{json.dumps(s['config'])}`",
            "",
            "## Alerts",
            "",
        ]
        L += [f"- ⚠️ {a}" for a in s["alerts"]] or ["- none"]
        L += ["", "## Items", ""]
        for b, it in s["items"].items():
            eta = f", ETA ~{it['eta_minutes']} min" if it["eta_minutes"] else ""
            L.append(f"- **{b}**: {it['done']}/{it['total']} done "
                     f"(avg {it['avg_seconds']}s/item{eta})")
        c = s["calls"]
        L += ["", "## Model calls", "",
              f"- {c['done']} done | {c['errors']} errors | **{c['truncated']} truncated** | "
              f"{c['output_tokens']:,} output tokens",
              f"- avg {c['avg_seconds']}s/call, max {c['max_seconds']}s, {c['avg_tok_s']} tok/s"]
        r = s["results"]
        if "zebra" in r:
            z = r["zebra"]
            L += ["", f"## ZebraLogic A/B/C (live, n={z['n']})", "",
                  f"- A read-puzzle: **{z['A_full']}** full (cell {z['A_cell']})",
                  f"- B clean-facts: **{z['B_full']}** full (cell {z['B_cell']})",
                  f"- C extraction recall: **{z['C_recall']}**",
                  f"- rows with any truncation: {z['truncated_rows']}", "",
                  "| size | n | A_full | B_full | C_recall |", "|---|---|---|---|---|"]
            L += [f"| {sz} | {g['n']} | {g['A_full']} | {g['B_full']} | {g['C_recall']} |"
                  for sz, g in z["by_size"].items()]
        if "referee" in r:
            f = r["referee"]
            L += ["", f"## Referee (live, n={f['n']})", "",
                  f"- raw: **{f['raw_full']}** -> refereed: **{f['ref_full']}** (lift {f['lift']:+})",
                  f"- converged {f['converged']}, avg attempts {f['avg_attempts']}", "",
                  "| size | n | raw | refereed |", "|---|---|---|---|"]
            L += [f"| {sz} | {g['n']} | {g['raw_full']} | {g['ref_full']} |"
                  for sz, g in f["by_size"].items()]
        if "tcp" in r:
            t = r["tcp"]
            L += ["", f"## TCP A/B/C (live, n={t['n']})", "",
                  f"- A read-dialogue: **{t['A']}** | B clean-facts: **{t['B']}**",
                  "- C fields: " + ", ".join(f"{k}={v}" for k, v in t["C_fields"].items())]
        L += ["", "## Recent calls", "", "| at | label | sec | out_tokens | trunc |", "|---|---|---|---|---|"]
        L += [f"| {x['at'][11:19]} | {x['label']} | {x['seconds']} | {x['output_tokens']} | "
              f"{'YES' if x['truncated'] else ''} |" for x in reversed(s["recent_calls"])]
        if s["recent_errors"]:
            L += ["", "## Recent errors", ""]
            L += [f"- `{x['at']}` **{x['label']}**: {x['error']}" for x in reversed(s["recent_errors"])]
        return "\n".join(L) + "\n"
