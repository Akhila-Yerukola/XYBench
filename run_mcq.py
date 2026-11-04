"""Multiple-choice preference test.

    python run_mcq.py --subset stackoverflow --model openai/gpt-5

Each query carries two short candidate responses (``mcq_variants``):
``stated_response`` (answers the stated request only) and ``hybrid_response``
(answers the stated request, identifies the misconception, and addresses the
intended request). The model is shown the query and both candidates in random
order and asked which is more helpful.

Output: outputs/{subset}/mcq_choices.{model}.jsonl and a tally on stdout.
Call settings are those used for the paper (see ``MCQ_KW``).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from prompts import PROMPT_MCQ_PREFERENCE, fill
from utils import SUBSETS, add_common_args, append_jsonl, complete, gather_limited, iter_jsonl, load_queries, mcq_path, opt_path, run_async

COMPARISONS = {
    "stated_vs_hybrid": ("stated_response", "hybrid_response"),
}

# Call settings used for the paper: temperature 0.7, extended thinking / reasoning
# effort "medium" (Claude thinking budget 5000; sets temperature to 1.0 for Claude).
MCQ_KW = dict(temperature=0.7, reasoning_effort="medium", verbosity="medium")


class Choice(BaseModel):
    chosen_response: str
    explanation: str = ""


async def _one(q: dict, model: str, comparison: str, rng: random.Random) -> dict:
    a_key, b_key = COMPARISONS[comparison]
    order = [a_key, b_key] if rng.random() < 0.5 else [b_key, a_key]
    v = q["mcq_variants"]
    prompt = fill(PROMPT_MCQ_PREFERENCE, {"user_query": q["query"], "response_a": v[order[0]], "response_b": v[order[1]]})
    row = {"id": q["id"], "model": model, "comparison": comparison, "order": order, "chosen": None, "reasoning": None, "error": None}
    try:
        c = await complete(model, prompt, response_model=Choice, **MCQ_KW)
        row["chosen"] = order[0] if "A" in c.chosen_response.upper() else order[1]
        row["reasoning"] = c.explanation
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)[:500]
    return row


async def run_mcq(subset: str, model: str, *, out_dir: Optional[Path], data_dir: Optional[Path], concurrency: int,
                  seed: int, limit: Optional[int]) -> Path:
    path = mcq_path(subset, model, out_dir)
    queries = [q for q in load_queries(subset, data_dir, limit=limit) if q.get("mcq_variants")]
    done = {(r["id"], r["comparison"]) for r in iter_jsonl(path)} if path.exists() else set()
    rng = random.Random(seed)
    jobs = [(q, c) for q in queries for c in COMPARISONS if (q["id"], c) not in done]
    print(f"[mcq] {subset} / {model}: {len(jobs)} comparisons to do, {len(done)} already in {path}")
    chunk = max(concurrency * 5, 20)
    for i in range(0, len(jobs), chunk):
        batch = jobs[i:i + chunk]
        results = await gather_limited([_one(q, model, c, rng) for q, c in batch], concurrency,
                                       desc=f"mcq {i // chunk + 1}/{(len(jobs) + chunk - 1) // chunk}")
        for r in results:
            if not isinstance(r, Exception):
                append_jsonl(path, r)
    return path


def tally(paths: list[Path]) -> dict[tuple[str, str], Counter]:
    out: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for p in paths:
        for r in iter_jsonl(p):
            out[(r["model"], r["comparison"])][r["chosen"] or "error"] += 1
    return out


def tally_markdown(t: dict[tuple[str, str], Counter]) -> str:
    lines = ["| Model | Comparison | stated | hybrid | errors | N |", "|---|---|---|---|---|---|"]
    for (model, comp), c in sorted(t.items()):
        n = sum(c.values())
        pct = lambda k: f"{100 * c.get(k, 0) / n:.1f}%" if n else "–"  # noqa: E731
        lines.append(f"| {model} | {comp} | {pct('stated_response')} | {pct('hybrid_response')} | {c.get('error', 0)} | {n} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, choices=SUBSETS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    add_common_args(ap)
    a = ap.parse_args()
    p = run_async(run_mcq(a.subset, a.model, out_dir=opt_path(a.out_dir), data_dir=opt_path(a.data_dir),
                          concurrency=a.concurrency, seed=a.seed, limit=a.limit))
    print(tally_markdown(tally([p])))


if __name__ == "__main__":
    main()
