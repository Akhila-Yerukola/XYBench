"""Sanity checks on the released data files."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_results import METRICS, human_rows  # noqa: E402
from utils import SUBSETS, human_reference_path, queries_path, read_jsonl  # noqa: E402

try:  # these tests need the dataset (Hugging Face Hub or --data-dir / XYBENCH_DATA_DIR)
    queries_path(SUBSETS[0])
except Exception as e:  # noqa: BLE001
    pytest.skip(f"dataset not available: {type(e).__name__}", allow_module_level=True)

EXPECTED_N ={"stackoverflow": 1557, "stackexchange": 184, "manually_curated": 102, "wikihow": 6268}
REQUIRED = ["id", "subset", "domain", "query", "stated_request", "intended_request", "stated_solution",
            "intended_solution", "misconception_identification", "misconception_mode", "source"]


@pytest.mark.parametrize("subset", SUBSETS)
def test_queries(subset):
    qs = read_jsonl(queries_path(subset))
    assert len(qs) == EXPECTED_N[subset]
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert all(k in q for k in REQUIRED), q["id"]
        for k in ("query", "stated_request", "intended_request", "stated_solution", "intended_solution", "misconception_identification"):
            assert q[k].strip(), (q["id"], k)
        assert (q.get("gap") is not None) == (subset == "wikihow")
        assert (q.get("misconception_type") is not None) == (subset in ("stackoverflow", "stackexchange"))
        assert q["misconception_mode"] == ("mistake_practice" if subset == "wikihow" else "request_mismatch")
        assert (q.get("mistake") is not None and q.get("correct_practice") is not None) == (subset == "wikihow")
        if q.get("mcq_variants"):
            assert set(q["mcq_variants"]) == {"stated_response", "hybrid_response"}


@pytest.mark.parametrize("subset", ["stackoverflow", "stackexchange", "manually_curated"])
def test_human_reference(subset):
    ids = {q["id"] for q in read_jsonl(queries_path(subset))}
    p = human_reference_path(subset)
    assert p is not None
    hs = read_jsonl(p)
    assert hs and all(h["id"] in ids for h in hs)
    assert all(h["answer_text"] for h in hs)
    assert all(set(h["metrics"]) >= {"s_solution_present", "i_solution_present", "emphasis", "misconception_identified"} for h in hs)


def test_human_rows_compute():
    for subset in ("stackoverflow", "stackexchange", "manually_curated"):
        rows = human_rows(subset, None)
        assert [r["system"] for r in rows] == ["human_all", "human_accepted", "human_highest"]
        all_row = rows[0]["cells"]
        assert all(all_row[m][1] > 0 for m in METRICS)
