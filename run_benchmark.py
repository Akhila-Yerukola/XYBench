"""Run several models on a subset end to end: generate responses, judge them, score.

    python run_benchmark.py --subset stackoverflow --models openai/gpt-5 anthropic/claude-sonnet-4-6 openai/gpt-4.1
    python run_benchmark.py --subset wikihow stackexchange --models openai/gpt-5 --condition query_only oracle_intent --limit 50
    python run_benchmark.py --subset manually_curated --models openai/gpt-5 --steps generate judge mcq score

Steps (``--steps``, default: generate judge score):
  generate   generate_responses.py for every (model, condition); models run in parallel,
             each with --concurrency requests in flight; one automatic pass over failures
  judge      evaluate_responses.py on each responses file, one model at a time (all judge
             calls go to the same provider); one automatic pass over failures
  mcq        run_mcq.py for every model
  score      compute_results.py for the subsets: prints the table, writes outputs/summary.md

Every step resumes from existing output files, so re-running the same command only
does the missing work. Generation settings are the defaults of generate_responses.py
unless overridden with --temperature / --max-tokens / --reasoning-effort / --verbosity.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
from pathlib import Path

from compute_results import compute, to_markdown
from evaluate_responses import EVALUATIONS, evaluate
from generate_responses import generate, resolve_settings
from run_mcq import run_mcq, tally, tally_markdown
from utils import CONDITIONS, DEFAULT_JUDGE, OUTPUT_DIR, SUBSETS, add_common_args, opt_path, responses_path, run_async

STEPS = ["generate", "judge", "mcq", "score"]


async def _gather(coros):
    """gather() must be called inside the running loop."""
    return await asyncio.gather(*coros)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", nargs="+", required=True, choices=SUBSETS)
    ap.add_argument("--models", nargs="+", required=True, help="litellm model ids")
    ap.add_argument("--condition", nargs="+", default=["query_only"], choices=CONDITIONS)
    ap.add_argument("--steps", nargs="+", default=["generate", "judge", "score"], choices=STEPS)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    ap.add_argument("--verbosity", choices=["none", "low", "medium", "high"])
    ap.add_argument("--keep-indentation", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="A/B order seed for the mcq step")
    add_common_args(ap)
    a = ap.parse_args()
    out_dir, data_dir = opt_path(a.out_dir), opt_path(a.data_dir)
    steps = set(a.steps)

    for subset in a.subset:
        print(f"\n===== {subset}: {len(a.models)} model(s) x {len(a.condition)} condition(s) =====")

        if "generate" in steps:
            for retry in (False, True):
                jobs = []
                for model in a.models:
                    gen = {**resolve_settings(model, a), "keep_indentation": a.keep_indentation,
                           "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
                    for cond in a.condition:
                        jobs.append(generate(subset, model, cond, out_dir=out_dir, data_dir=data_dir, concurrency=a.concurrency,
                                             limit=a.limit, ids=None, retry_failed=retry, gen=gen))
                run_async(_gather(jobs))

        if "judge" in steps:
            for retry in (False, True):
                for model in a.models:
                    for cond in a.condition:
                        rp = responses_path(subset, model, cond, out_dir)
                        if not rp.exists():
                            print(f"[judge] skip {rp} (not found)")
                            continue
                        run_async(evaluate(subset, rp, judge=a.judge, out_dir=out_dir, data_dir=data_dir, concurrency=a.concurrency,
                                           evaluations=set(EVALUATIONS), limit=None, retry_failed=retry))

        if "mcq" in steps:
            paths = run_async(_gather([run_mcq(subset, m, out_dir=out_dir, data_dir=data_dir, concurrency=a.concurrency,
                                               seed=a.seed, limit=a.limit) for m in a.models]))
            print(tally_markdown(tally(list(paths))))

    if "score" in steps:
        od = out_dir or OUTPUT_DIR
        rows = compute(a.subset, out_dir=od, data_dir=data_dir, judge=a.judge, include_human=True)
        text = to_markdown(rows)
        print(text)
        od.mkdir(parents=True, exist_ok=True)
        (od / "summary.md").write_text(text)
        print(f"\nwrote {od / 'summary.md'}")


if __name__ == "__main__":
    main()
