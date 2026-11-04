"""Shared helpers: file layout, JSONL I/O, model naming, and LLM calls."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Type, TypeVar

import json_repair
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

ROOT = Path(__file__).resolve().parent
# Benchmark data: downloaded from this Hugging Face dataset (cached by huggingface_hub)
# unless a local directory with {subset}/queries.jsonl is given (--data-dir or XYBENCH_DATA_DIR).
DATASET = os.environ.get("XYBENCH_DATASET", "allenai/XYBench-dataset")
DATA_DIR = Path(os.environ["XYBENCH_DATA_DIR"]) if os.environ.get("XYBENCH_DATA_DIR") else None
OUTPUT_DIR = Path(os.environ.get("XYBENCH_OUTPUT_DIR", ROOT / "outputs"))

SUBSETS = ["stackoverflow", "stackexchange", "manually_curated", "wikihow"]
SUBSET_DISPLAY = {
    "stackoverflow": "Stack Overflow",
    "stackexchange": "Stack Exchange",
    "manually_curated": "Manually Curated",
    "wikihow": "WikiHow",
}
CONDITIONS = ["query_only", "oracle_intent", "clear_misconceptions"]
DEFAULT_JUDGE = "anthropic/claude-sonnet-4-5-20250929"

MODEL_DISPLAY = {
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "openai/gpt-5": "GPT-5",
    "openai/gpt-4.1": "GPT-4.1",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "together_ai/Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "together_ai/deepseek-ai/DeepSeek-V4-Pro": "DeepSeek-V4 Pro",
}


def model_display(name: str) -> str:
    return MODEL_DISPLAY.get(name, name.split("/")[-1])


def model_slug(name: str) -> str:
    """File-name safe model id (``/`` -> ``__``)."""
    return name.replace("/", "__")


# ── File layout ──────────────────────────────────────────────────────────────

def _data_file(subset: str, name: str, data_dir: Path | None) -> Optional[Path]:
    """Local path of data/{subset}/{name}: from ``data_dir`` if given, else downloaded
    from the Hugging Face dataset. None if the file does not exist there."""
    d = data_dir or DATA_DIR
    if d is not None:
        p = Path(d) / subset / name
        return p if p.exists() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        return Path(hf_hub_download(repo_id=DATASET, filename=f"{subset}/{name}", repo_type="dataset"))
    except EntryNotFoundError:
        return None


def queries_path(subset: str, data_dir: Path | None = None) -> Path:
    p = _data_file(subset, "queries.jsonl", data_dir)
    if p is None:
        raise FileNotFoundError(f"{subset}/queries.jsonl not found in {data_dir or DATA_DIR or DATASET}")
    return p


def human_reference_path(subset: str, data_dir: Path | None = None) -> Optional[Path]:
    """None when the subset has no human reference answers (wikihow)."""
    return _data_file(subset, "human_reference.jsonl", data_dir)


def responses_path(subset: str, model: str, condition: str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUTPUT_DIR) / subset / f"responses.{model_slug(model)}.{condition}.jsonl"


def judgments_path(subset: str, model: str, condition: str, judge: str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUTPUT_DIR) / subset / f"judgments.{model_slug(model)}.{condition}.{model_slug(judge)}.jsonl"


def scores_path(subset: str, model: str, condition: str, judge: str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUTPUT_DIR) / subset / f"scores.{model_slug(model)}.{condition}.{model_slug(judge)}.jsonl"


def mcq_path(subset: str, model: str, out_dir: Path | None = None) -> Path:
    return (out_dir or OUTPUT_DIR) / subset / f"mcq_choices.{model_slug(model)}.jsonl"


# ── JSONL ────────────────────────────────────────────────────────────────────

def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(subset: str, data_dir: Path | None = None, ids: Optional[set] = None, limit: Optional[int] = None) -> list[dict]:
    qs = read_jsonl(queries_path(subset, data_dir))
    if ids is not None:
        qs = [q for q in qs if q["id"] in ids]
    if limit:
        qs = qs[:limit]
    return qs


# ── LLM calls (litellm) ──────────────────────────────────────────────────────
#
# API keys come from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY,
# GEMINI_API_KEY, TOGETHERAI_API_KEY, ...); a .env file in the repository root
# or working directory is loaded if present.

def load_dotenv() -> None:
    for p in (Path(".env"), ROOT / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

_RETRIABLE = ("rate_limit", "ratelimit", "overloaded", "timeout", "temporarily", "503", "529")


def _provider_kwargs(model: str, reasoning_effort: Optional[str], verbosity: Optional[str]) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if "claude" in model:
        if reasoning_effort in (None, "none"):
            kw["thinking"] = {"type": "disabled"}
        else:
            budget = {"low": 2000, "medium": 5000, "high": 10000}.get(reasoning_effort, 5000)
            kw["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        if reasoning_effort not in (None, "none"):
            kw["reasoning_effort"] = reasoning_effort
        if verbosity:
            kw["verbosity"] = verbosity
    return kw


def default_max_tokens(model: str, requested: Optional[int]) -> int:
    """``requested`` wins; otherwise a per-model-family default (reasoning
    models need room for their reasoning tokens)."""
    if requested is not None:
        return requested
    if "Qwen3.5" in model:
        return 20000   # reasoning tokens count against the budget
    if "gpt-5" in model or "claude" in model:
        return 10000
    return 8000


def parse_json_dict(text: str) -> dict:
    obj = json_repair.loads(text)
    if isinstance(obj, dict) and set(obj) == {"properties"}:
        obj = obj["properties"]
    if not isinstance(obj, dict):
        raise ValueError(f"could not parse a JSON object from: {text[:200]}")
    return obj


async def complete(
    model: str,
    prompt: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    response_model: Optional[Type[T]] = None,
    retries: int = 3,
) -> str | T:
    """One chat completion. With ``response_model`` the output is parsed into
    that pydantic model (structured output is requested where supported)."""
    from litellm import acompletion

    kw = _provider_kwargs(model, reasoning_effort, verbosity)
    if "claude" in model and kw.get("thinking", {}).get("type") == "enabled":
        temperature = 1.0  # required when thinking is enabled
    if temperature is not None:
        kw["temperature"] = temperature
    if response_model is not None and "together" not in model:
        kw["response_format"] = response_model

    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=default_max_tokens(model, max_tokens),
                drop_params=True,
                **kw,
            )
            text = resp.choices[0].message.content
            if text is None:
                raise ValueError(f"empty content (finish_reason={resp.choices[0].finish_reason})")
            if response_model is None:
                return text
            return response_model.model_validate_json(json.dumps(parse_json_dict(text)))
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e).lower()
            await asyncio.sleep((2 ** attempt) * 5 if any(t in msg for t in _RETRIABLE) else 10)
    raise RuntimeError(f"{model}: failed after {retries} attempts: {last}")


async def gather_limited(coros, concurrency: int, desc: str = ""):
    """Run coroutines with a concurrency limit and a progress bar; an exception
    is returned in place of its result."""
    from tqdm.asyncio import tqdm

    sem = asyncio.Semaphore(concurrency)

    async def run(c):
        async with sem:
            try:
                return await c
            except Exception as e:  # noqa: BLE001
                return e

    return await tqdm.gather(*(run(c) for c in coros), desc=desc)


async def _close_clients() -> None:
    try:
        import litellm
        fn = getattr(litellm, "close_litellm_async_clients", None)
        if fn is not None:
            await fn()
    except Exception:  # noqa: BLE001
        pass
    await asyncio.sleep(0.1)


def run_async(coro):
    async def wrapped():
        try:
            return await coro
        finally:
            await _close_clients()
    return asyncio.run(wrapped())


def add_common_args(ap, *, out_dir=True) -> None:
    ap.add_argument("--data-dir", help=f"local directory with {{subset}}/queries.jsonl (default: download from the "
                                        f"Hugging Face dataset {DATASET})")
    if out_dir:
        ap.add_argument("--out-dir", help="output directory (default: ./outputs)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int, help="only process the first N queries")


def opt_path(p: Optional[str]) -> Optional[Path]:
    return Path(p) if p else None
