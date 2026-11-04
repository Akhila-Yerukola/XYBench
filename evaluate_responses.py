"""Evaluate model responses with the LLM judge.

    python evaluate_responses.py --subset stackoverflow --responses outputs/stackoverflow/responses.openai__gpt-5.query_only.jsonl

Three evaluations per response, each driven by the query's metadata:

  solution_presence            extract the answer units of the response, then tag each
                               direct (matches stated_solution) / alternate (matches
                               intended_solution) / other
  misconception_identification tag each answer unit
                               stackoverflow / stackexchange / manually_curated: stated_goal_discouraged /
                                   stated_goal_entertained / intended_goal / other (from stated_request, intended_request)
                               wikihow: mistake_identified / correct_practice / other (from mistake, correct_practice)
  emphasis                     approach_1 (organized around stated_solution) / approach_2
                               (around intended_solution) / balanced / neither

Default judge: Claude Sonnet 4.5 (thinking disabled, temperature 0.7, JSON output).

Output: outputs/{subset}/judgments.{model}.{condition}.{judge}.jsonl with the
three blocks above (null when an evaluation failed; see ``errors``). Runs
resume from the output file; ``--retry-failed`` re-runs records with errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from prompts import PROMPT_EMPHASIS, PROMPT_MISCONCEPTION, PROMPT_PRESENCE_EXTRACT, PROMPT_PRESENCE_TAG, fill
from utils import (
    DEFAULT_JUDGE,
    SUBSETS,
    add_common_args,
    append_jsonl,
    complete,
    gather_limited,
    iter_jsonl,
    judgments_path,
    load_queries,
    opt_path,
    read_jsonl,
    run_async,
    write_jsonl,
)

EVALUATIONS = ["solution_presence", "misconception_identification", "emphasis"]


class TaggedUnit(BaseModel):
    text: str
    explanation: str
    label: str


class Extracted(BaseModel):
    answer_units: List[str]


class Tagged(BaseModel):
    answer_units: List[TaggedUnit]


class Alignment(BaseModel):
    explanation_alignment_with_approach_1: str
    explanation_alignment_with_approach_2: str
    explanation_alignment_with_balanced: str
    final_response_alignment: str
    justification: str


# Judge call settings used for the paper: no extended thinking, temperature 0.7.
JUDGE_KW = dict(reasoning_effort=None, verbosity=None, temperature=0.7)


# ── Prompt construction from query metadata ──────────────────────────────────

def presence_extract_prompt(response: str) -> str:
    return fill(PROMPT_PRESENCE_EXTRACT, {"response": response})


def presence_tag_prompt(q: dict, units: list[str]) -> str:
    return fill(PROMPT_PRESENCE_TAG, {
        "answer_units": json.dumps(units),
        "direct_literal_answer": q["stated_solution"],
        "better_alternatives": q["intended_solution"].replace("Better alternatives ", "Alternatives "),
    })


def misconception_prompt(q: dict, units: list[str]) -> str:
    mode = q["misconception_mode"]
    if mode == "mistake_practice":
        values = {
            "where_they_went_wrong": "Explains how the user might have done wrong: " + q["mistake"],
            "solution_to_y": "Addresses or focuses on the specific approach or solution: " + q["correct_practice"],
        }
    else:
        s, i = q["stated_request"], q["intended_request"]
        values = {
            "stated_goal_discouraged": f'Explains why "{s}" is either not possible, incorrect, or misguided and won\'t solve the problem. ',
            "stated_goal_entertained": f'Provides a solution or workaround that directly addresses "{s}", without questioning it',
            "intended_goal": f'Addresses or focuses on "{i}" but different and NOT about "{s}". ',
        }
    return fill(PROMPT_MISCONCEPTION[mode], {"answer_units": json.dumps(units), **values})


def emphasis_prompt(q: dict, response: str) -> str:
    ss, is_ = q["stated_solution"], q["intended_solution"]
    approach_1 = ss.replace("A direct answer would ", "An answer that focuses on: ") if "A direct answer would " in ss else "An answer that focuses on: " + ss
    if q["subset"] == "wikihow":
        approach_2 = is_
    else:
        approach_2 = is_.replace("Better alternatives ", "An answer that focuses on: ") if "Better alternatives " in is_ else "An answer that focuses on: " + is_
    return fill(PROMPT_EMPHASIS, {"response": response, "approach_1": approach_1, "approach_2": approach_2})


# ── Judging ──────────────────────────────────────────────────────────────────

async def judge_one(q: dict, r: dict, judge: str, evaluations: set[str]) -> dict:
    out = {"id": q["id"], "model": r["model"], "condition": r["condition"], "judge": judge,
           "solution_presence": None, "misconception_identification": None, "emphasis": None, "errors": {}}
    response = r.get("response")
    if not response:
        out["errors"]["response"] = "no response to judge"
        return out

    units: Optional[list[str]] = None
    if evaluations & {"solution_presence", "misconception_identification"}:
        try:
            units = (await complete(judge, presence_extract_prompt(response), response_model=Extracted, **JUDGE_KW)).answer_units
        except Exception as e:  # noqa: BLE001
            out["errors"]["solution_presence"] = f"extraction failed: {e}"[:500]
        if units is not None and "solution_presence" in evaluations:
            try:
                tg = await complete(judge, presence_tag_prompt(q, units), response_model=Tagged, **JUDGE_KW)
                out["solution_presence"] = {"answer_units": units, "tagged": [u.model_dump() for u in tg.answer_units]}
            except Exception as e:  # noqa: BLE001
                out["errors"]["solution_presence"] = f"tagging failed: {e}"[:500]
        if "misconception_identification" in evaluations:
            if units is None:
                out["errors"]["misconception_identification"] = "skipped: extraction failed"
            else:
                try:
                    tg = await complete(judge, misconception_prompt(q, units), response_model=Tagged, **JUDGE_KW)
                    out["misconception_identification"] = {"tagged": [u.model_dump() for u in tg.answer_units]}
                except Exception as e:  # noqa: BLE001
                    out["errors"]["misconception_identification"] = f"tagging failed: {e}"[:500]

    if "emphasis" in evaluations:
        try:
            al = await complete(judge, emphasis_prompt(q, response), response_model=Alignment, **JUDGE_KW)
            out["emphasis"] = al.model_dump()
        except Exception as e:  # noqa: BLE001
            out["errors"]["emphasis"] = f"alignment failed: {e}"[:500]
    return out


async def evaluate(subset: str, responses: Path, *, judge: str, out_dir: Optional[Path], data_dir: Optional[Path],
                   concurrency: int, evaluations: set[str], limit: Optional[int], retry_failed: bool) -> Path:
    rs = read_jsonl(responses)
    if not rs:
        raise SystemExit(f"no responses in {responses}")
    model, condition = rs[0]["model"], rs[0]["condition"]
    path = judgments_path(subset, model, condition, judge, out_dir)
    queries = {q["id"]: q for q in load_queries(subset, data_dir)}

    done = {row["id"]: row for row in iter_jsonl(path)} if path.exists() else {}
    if retry_failed:
        done = {k: v for k, v in done.items() if not v.get("errors")}
        write_jsonl(path, done.values())
    todo = [r for r in rs if r["id"] not in done and r["id"] in queries]
    if limit:
        todo = todo[:limit]
    print(f"[evaluate] {subset} / {model} / {condition} / judge={judge}: {len(todo)} to do, {len(done)} already in {path}")

    chunk = max(concurrency * 5, 20)
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        results = await gather_limited([judge_one(queries[r["id"]], r, judge, evaluations) for r in batch], concurrency,
                                       desc=f"evaluate {i // chunk + 1}/{(len(todo) + chunk - 1) // chunk}")
        for j in results:
            if not isinstance(j, Exception):
                append_jsonl(path, j)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, choices=SUBSETS)
    ap.add_argument("--responses", nargs="+", required=True, help="responses jsonl file(s) from generate_responses.py")
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--evaluations", nargs="+", default=EVALUATIONS, choices=EVALUATIONS)
    ap.add_argument("--retry-failed", action="store_true", help="re-run records that have errors")
    add_common_args(ap)
    a = ap.parse_args()
    for f in a.responses:
        run_async(evaluate(a.subset, Path(f), judge=a.judge, out_dir=opt_path(a.out_dir), data_dir=opt_path(a.data_dir),
                           concurrency=a.concurrency, evaluations=set(a.evaluations), limit=a.limit, retry_failed=a.retry_failed))


if __name__ == "__main__":
    main()
