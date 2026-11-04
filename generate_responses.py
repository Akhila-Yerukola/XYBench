"""Generate model responses to XYBench queries.

    python generate_responses.py --subset stackoverflow --model openai/gpt-5
    python generate_responses.py --subset wikihow --model anthropic/claude-sonnet-4-6 --condition query_only oracle_intent

Conditions:
  query_only            the query as-is (default)
  oracle_intent         the query followed by the stated request and the intended request
  clear_misconceptions  the query followed by "please clear any misconceptions in the request"

Generation settings (``DEFAULT_SETTINGS``), the same for every subset:

  temperature 1.0
  reasoning effort "medium"   Claude: extended thinking with a 5000-token budget
                              (temperature is then fixed at 1.0); GPT-5 / Gemini:
                              medium reasoning
  verbosity "medium"          GPT-5 only
  max tokens                  10000 for GPT-5 and Claude, 20000 for Qwen3.5
                              (reasoning tokens count against it), 8000 otherwise

Parameters a provider does not support are dropped. ``--temperature``,
``--max-tokens``, ``--reasoning-effort`` and ``--verbosity`` override the
defaults; the settings actually used are stored in each record's ``generation``.

Output: outputs/{subset}/responses.{model}.{condition}.jsonl, one record per query
(``id, model, condition, prompt, response, generation``). Runs resume from the
output file; ``--retry-failed`` re-runs queries whose response is null.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

from prompts import build_query_prompt, normalize_whitespace
from utils import (
    CONDITIONS,
    SUBSETS,
    add_common_args,
    append_jsonl,
    complete,
    default_max_tokens,
    gather_limited,
    iter_jsonl,
    load_queries,
    opt_path,
    responses_path,
    run_async,
    write_jsonl,
)

# Default generation settings, identical for every subset. Max tokens depends on
# the model family (see utils.default_max_tokens).
DEFAULT_SETTINGS = {"temperature": 1.0, "reasoning_effort": "medium", "verbosity": "medium"}


def resolve_settings(model: str, a) -> dict:
    """Default settings, overridden by any explicit command-line flags."""
    none = lambda v: None if v in (None, "none") else v  # noqa: E731
    return {
        "temperature": DEFAULT_SETTINGS["temperature"] if a.temperature is None else a.temperature,
        "max_tokens": default_max_tokens(model, a.max_tokens),
        "reasoning_effort": DEFAULT_SETTINGS["reasoning_effort"] if a.reasoning_effort is None else none(a.reasoning_effort),
        "verbosity": DEFAULT_SETTINGS["verbosity"] if a.verbosity is None else none(a.verbosity),
    }


async def _one(q: dict, model: str, condition: str, gen: dict) -> dict:
    prompt = build_query_prompt(q["subset"], condition, q["query"], q["stated_request"], q["intended_request"])
    if not gen["keep_indentation"]:
        prompt = normalize_whitespace(prompt)
    row = {"id": q["id"], "model": model, "condition": condition, "prompt": prompt, "response": None, "generation": gen}
    try:
        row["response"] = await complete(
            model, prompt,
            temperature=gen["temperature"], max_tokens=gen["max_tokens"],
            reasoning_effort=gen["reasoning_effort"], verbosity=gen["verbosity"],
        )
    except Exception as e:  # noqa: BLE001
        row["generation"] = {**gen, "error": str(e)[:500]}
    return row


async def generate(subset: str, model: str, condition: str, *, out_dir: Optional[Path], data_dir: Optional[Path],
                   concurrency: int, limit: Optional[int], ids: Optional[set], retry_failed: bool, gen: dict) -> Path:
    path = responses_path(subset, model, condition, out_dir)
    queries = load_queries(subset, data_dir, ids, limit)

    done = set()
    if path.exists():
        rows = list(iter_jsonl(path))
        if retry_failed:
            rows = [r for r in rows if r.get("response")]
            write_jsonl(path, rows)
        done = {r["id"] for r in rows}
    todo = [q for q in queries if q["id"] not in done]
    print(f"[generate] {subset} / {model} / {condition}: {len(todo)} to do, {len(done)} already in {path}")

    chunk = max(concurrency * 5, 20)
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        results = await gather_limited([_one(q, model, condition, gen) for q in batch], concurrency,
                                       desc=f"generate {i // chunk + 1}/{(len(todo) + chunk - 1) // chunk}")
        ok = 0
        for r in results:
            if isinstance(r, Exception):
                continue
            append_jsonl(path, r)
            ok += int(r["response"] is not None)
        print(f"  {ok}/{len(batch)} succeeded")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, choices=SUBSETS)
    ap.add_argument("--model", required=True, help="litellm model id, e.g. openai/gpt-5, anthropic/claude-sonnet-4-6")
    ap.add_argument("--condition", nargs="+", default=["query_only"], choices=CONDITIONS)
    ap.add_argument("--temperature", type=float, help="default: 1.0")
    ap.add_argument("--max-tokens", type=int, help="default: 10000 for GPT-5 and Claude, 20000 for Qwen3.5, 8000 otherwise")
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"],
                    help="extended thinking / reasoning effort (default: medium)")
    ap.add_argument("--verbosity", choices=["none", "low", "medium", "high"],
                    help="GPT-5 verbosity (default: medium)")
    ap.add_argument("--ids", nargs="*", help="only these query ids")
    ap.add_argument("--retry-failed", action="store_true", help="re-run queries whose response is null")
    ap.add_argument("--keep-indentation", action="store_true",
                    help="send the query verbatim (by default leading whitespace on each line is stripped)")
    add_common_args(ap)
    a = ap.parse_args()

    gen = {
        **resolve_settings(a.model, a),
        "keep_indentation": a.keep_indentation,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[generate] settings: {gen}")
    for cond in a.condition:
        run_async(generate(a.subset, a.model, cond, out_dir=opt_path(a.out_dir), data_dir=opt_path(a.data_dir),
                           concurrency=a.concurrency, limit=a.limit, ids=set(a.ids) if a.ids else None,
                           retry_failed=a.retry_failed, gen=gen))


if __name__ == "__main__":
    main()
