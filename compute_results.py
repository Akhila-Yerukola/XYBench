"""Compute metrics and the overall results table.

    python compute_results.py                     # all subsets, markdown to stdout + outputs/summary.md
    python compute_results.py --format latex --subset stackoverflow stackexchange

Reads every judgments file under outputs/{subset}/ (for the given judge),
writes outputs/{subset}/scores.*.jsonl, and prints one row per
(model, condition) plus the human reference rows from data/{subset}/human_reference.jsonl.

Per-response metrics (from the three judge evaluations):
  s_solution_present        solution_presence has a unit labelled "direct"
  i_solution_present        solution_presence has a unit labelled "alternate"
  emphasis                  approach_1 -> s_solution, approach_2 -> i_solution, balanced,
                            neither (organized around none of the two; counted as its own
                            outcome, never as S-solution emphasis)
  misconception_identified  stackoverflow / stackexchange / manually_curated:
                              (discouraged AND intended) OR (intended AND NOT discouraged AND NOT entertained)
                            wikihow: "mistake_identified" present
  fully_pragmatic           misconception_identified AND i_solution_present AND emphasis == i_solution
  fully_cooperative         misconception_identified AND s_solution_present AND i_solution_present
                            AND emphasis in {balanced, i_solution}

Rates are over responses for which the metric is defined; N is shown per cell.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from utils import (
    DEFAULT_JUDGE,
    OUTPUT_DIR,
    SUBSET_DISPLAY,
    SUBSETS,
    human_reference_path,
    model_display,
    model_slug,
    opt_path,
    read_jsonl,
    scores_path,
    write_jsonl,
)

METRICS = ["s_solution_present", "i_solution_present", "emphasis_s_solution", "emphasis_balanced", "emphasis_i_solution",
           "emphasis_neither", "misconception_identified", "fully_pragmatic", "fully_cooperative"]
METRIC_DISPLAY = {
    "s_solution_present": "S-soln (present)", "i_solution_present": "I-soln (present)",
    "emphasis_s_solution": "Emph: S-soln", "emphasis_balanced": "Emph: balanced", "emphasis_i_solution": "Emph: I-soln",
    "emphasis_neither": "Emph: neither",
    "misconception_identified": "S-I-Miscon. identif.", "fully_pragmatic": "Fully Prag.", "fully_cooperative": "Fully Coop.",
}
EMPHASIS_MAP = {"approach_1": "s_solution", "approach_2": "i_solution", "balanced": "balanced", "neither": "neither"}
HUMAN_VARIANTS = {"human_all": "Human (all)", "human_accepted": "Human (accepted)", "human_highest": "Human (highest score)"}


# ── Per-response metrics ─────────────────────────────────────────────────────

def labels_of(block: Optional[dict]) -> Optional[set]:
    if not block:
        return None
    return {str(u.get("label") or "").strip().lower() for u in block.get("tagged", [])}


def compute_metrics(presence: Optional[dict], misconception: Optional[dict], emphasis: Optional[dict], mode: str) -> dict:
    m = {"s_solution_present": None, "i_solution_present": None, "emphasis": None,
         "misconception_identified": None, "fully_pragmatic": None, "fully_cooperative": None}
    L1 = labels_of(presence)
    if L1 is not None:
        m["s_solution_present"], m["i_solution_present"] = "direct" in L1, "alternate" in L1
    L2 = labels_of(misconception)
    if L2 is not None:
        if mode == "mistake_practice":
            m["misconception_identified"] = "mistake_identified" in L2
        else:
            d, en, it = "stated_goal_discouraged" in L2, "stated_goal_entertained" in L2, "intended_goal" in L2
            m["misconception_identified"] = (d and it) or (it and not d and not en)
    if emphasis:
        label = str(emphasis.get("final_response_alignment") or "").strip().lower()
        m["emphasis"] = EMPHASIS_MAP.get(label)  # unknown label -> undefined
    return add_composites(m)


def add_composites(m: dict) -> dict:
    if None not in (m["misconception_identified"], m["i_solution_present"], m["emphasis"]):
        m["fully_pragmatic"] = bool(m["misconception_identified"] and m["i_solution_present"] and m["emphasis"] == "i_solution")
        if m["s_solution_present"] is not None:
            m["fully_cooperative"] = bool(m["misconception_identified"] and m["s_solution_present"] and m["i_solution_present"]
                                          and m["emphasis"] in ("balanced", "i_solution"))
    return m


def score_judgment(j: dict, mode: str) -> dict:
    m = compute_metrics(j.get("solution_presence"), j.get("misconception_identification"), j.get("emphasis"), mode)
    return {"id": j["id"], "model": j["model"], "condition": j["condition"], **m}


# ── Aggregation ──────────────────────────────────────────────────────────────

def flags(m: dict) -> dict:
    e = m.get("emphasis")
    return {
        "s_solution_present": m.get("s_solution_present"),
        "i_solution_present": m.get("i_solution_present"),
        "emphasis_s_solution": None if e is None else e == "s_solution",
        "emphasis_balanced": None if e is None else e == "balanced",
        "emphasis_i_solution": None if e is None else e == "i_solution",
        "emphasis_neither": None if e is None else e == "neither",
        "misconception_identified": m.get("misconception_identified"),
        "fully_pragmatic": m.get("fully_pragmatic"),
        "fully_cooperative": m.get("fully_cooperative"),
    }


def aggregate(items) -> dict[str, tuple[Optional[float], int]]:
    n = {k: 0 for k in METRICS}
    k_ = {k: 0 for k in METRICS}
    for it in items:
        for key, v in flags(it).items():
            if v is None:
                continue
            n[key] += 1
            k_[key] += int(bool(v))
    return {key: ((k_[key] / n[key]) if n[key] else None, n[key]) for key in METRICS}


def human_variants(h: dict) -> list[str]:
    v = ["human_all"]
    if h.get("is_accepted"):
        v.append("human_accepted")
    if h.get("is_highest_score") or h.get("is_high_score"):
        v.append("human_highest")
    return v


def human_rows(subset: str, data_dir: Optional[Path]) -> list[dict]:
    p = human_reference_path(subset, data_dir)
    if p is None:
        return []
    buckets = {v: [] for v in HUMAN_VARIANTS}
    for h in read_jsonl(p):
        for v in human_variants(h):
            buckets[v].append(h["metrics"])
    return [{"subset": subset, "system": v, "condition": "query_only", "cells": aggregate(ms)} for v, ms in buckets.items()]


_JUDG_RE = re.compile(r"^judgments\.(?P<model>.+?)\.(?P<condition>query_only|oracle_intent|clear_misconceptions)\.(?P<judge>.+)\.jsonl$")


def model_rows(subset: str, out_dir: Path, data_dir: Optional[Path], judge: str) -> list[dict]:
    d = out_dir / subset
    if not d.exists():
        return []
    from utils import load_queries
    mode = {q["id"]: q["misconception_mode"] for q in load_queries(subset, data_dir)}
    rows = []
    for f in sorted(d.glob("judgments.*.jsonl")):
        m = _JUDG_RE.match(f.name)
        if not m or m.group("judge") != model_slug(judge):
            continue
        js = read_jsonl(f)
        if not js:
            continue
        scores = [score_judgment(j, mode.get(j["id"], "request_mismatch")) for j in js]
        write_jsonl(scores_path(subset, js[0]["model"], js[0]["condition"], judge, out_dir), scores)
        rows.append({"subset": subset, "system": js[0]["model"], "condition": js[0]["condition"], "cells": aggregate(scores)})
    return rows


# ── Rendering ────────────────────────────────────────────────────────────────

def display_name(system: str) -> str:
    return HUMAN_VARIANTS.get(system, model_display(system))


def to_markdown(rows: list[dict]) -> str:
    out = []
    by = defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    for subset, rs in by.items():
        out.append(f"\n### {SUBSET_DISPLAY[subset]}\n")
        out.append("| System | Condition | " + " | ".join(METRIC_DISPLAY[m] for m in METRICS) + " |")
        out.append("|---|---|" + "---|" * len(METRICS))
        for r in rs:
            vals = ["–" if v is None else f"{v:.2f} (n={n})" for v, n in (r["cells"][m] for m in METRICS)]
            out.append(f"| {display_name(r['system'])} | {r['condition']} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def to_latex(rows: list[dict]) -> str:
    out = ["\\begin{tabular}{ll" + "c" * len(METRICS) + "}", "\\toprule",
           "Source & Model & " + " & ".join(METRIC_DISPLAY[m] for m in METRICS) + " \\\\", "\\midrule"]
    by = defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    for subset, rs in by.items():
        for i, r in enumerate(rs):
            vals = ["–" if r["cells"][m][0] is None else f"{r['cells'][m][0]:.2f}" for m in METRICS]
            out.append(f"{SUBSET_DISPLAY[subset] if i == 0 else ''} & {display_name(r['system'])} & " + " & ".join(vals) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    return "\n".join(out)


def to_csv(rows: list[dict], path: Path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "system", "condition"] + METRICS + [f"{m}_n" for m in METRICS])
        for r in rows:
            w.writerow([r["subset"], r["system"], r["condition"]]
                       + [None if r["cells"][m][0] is None else round(r["cells"][m][0], 4) for m in METRICS]
                       + [r["cells"][m][1] for m in METRICS])


def compute(subsets, *, out_dir: Optional[Path], data_dir: Optional[Path], judge: str, include_human: bool) -> list[dict]:
    out_dir = out_dir or OUTPUT_DIR
    rows = []
    for s in subsets:
        rows += model_rows(s, out_dir, data_dir, judge)
        if include_human:
            rows += human_rows(s, data_dir)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", nargs="*", choices=SUBSETS, default=SUBSETS)
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="use judgments from this judge")
    ap.add_argument("--format", default="md", choices=["md", "csv", "latex"])
    ap.add_argument("--output", help="file to write (default: outputs/summary.{md,csv,tex})")
    ap.add_argument("--no-human", action="store_true", help="omit human reference rows")
    ap.add_argument("--data-dir", help="local directory with {subset}/queries.jsonl (default: the Hugging Face dataset)")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    out_dir = opt_path(a.out_dir) or OUTPUT_DIR
    rows = compute(a.subset, out_dir=out_dir, data_dir=opt_path(a.data_dir), judge=a.judge, include_human=not a.no_human)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.format == "csv":
        path = opt_path(a.output) or out_dir / "summary.csv"
        to_csv(rows, path)
    else:
        text = to_markdown(rows) if a.format == "md" else to_latex(rows)
        print(text)
        path = opt_path(a.output) or out_dir / ("summary.md" if a.format == "md" else "summary.tex")
        path.write_text(text)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
