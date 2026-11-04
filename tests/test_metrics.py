import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compute_results import aggregate, compute_metrics  # noqa: E402


def units(*labels):
    return [{"text": f"u{i}", "explanation": "", "label": l} for i, l in enumerate(labels)]


def e1(*labels):
    return {"answer_units": [f"u{i}" for i in range(len(labels))], "tagged": units(*labels)}


def e2(*labels):
    return {"tagged": units(*labels)}


def e3(label):
    return {"explanation_alignment_with_approach_1": "", "explanation_alignment_with_approach_2": "",
            "explanation_alignment_with_balanced": "", "final_response_alignment": label, "justification": ""}


def test_presence():
    m = compute_metrics(e1("direct", "other"), None, None, "request_mismatch")
    assert m["s_solution_present"] is True and m["i_solution_present"] is False
    m = compute_metrics(e1("alternate"), None, None, "request_mismatch")
    assert m["s_solution_present"] is False and m["i_solution_present"] is True
    m = compute_metrics(None, None, None, "request_mismatch")
    assert m["s_solution_present"] is None and m["fully_pragmatic"] is None


def test_misconception_request_mismatch():
    f = lambda *l: compute_metrics(None, e2(*l), None, "request_mismatch")["misconception_identified"]  # noqa: E731
    assert f("stated_goal_discouraged", "intended_goal") is True
    assert f("intended_goal") is True
    assert f("intended_goal", "stated_goal_entertained") is False
    assert f("stated_goal_discouraged", "intended_goal", "stated_goal_entertained") is True
    assert f("stated_goal_discouraged") is False
    assert f("other") is False


def test_misconception_mistake_practice():
    f = lambda *l: compute_metrics(None, e2(*l), None, "mistake_practice")["misconception_identified"]  # noqa: E731
    assert f("mistake_identified", "other") is True
    assert f("correct_practice") is False


def test_emphasis_and_composites():
    m = compute_metrics(e1("direct", "alternate"), e2("intended_goal"), e3("approach_2"), "request_mismatch")
    assert m["emphasis"] == "i_solution" and m["fully_pragmatic"] is True and m["fully_cooperative"] is True
    m = compute_metrics(e1("direct", "alternate"), e2("intended_goal"), e3("balanced"), "request_mismatch")
    assert m["fully_pragmatic"] is False and m["fully_cooperative"] is True
    m = compute_metrics(e1("alternate"), e2("intended_goal"), e3("approach_2"), "request_mismatch")
    assert m["fully_pragmatic"] is True and m["fully_cooperative"] is False
    m = compute_metrics(e1("alternate"), e2("intended_goal"), e3("approach_1"), "request_mismatch")
    assert m["emphasis"] == "s_solution" and m["fully_pragmatic"] is False


def test_aggregate_skips_undefined():
    cells = aggregate([compute_metrics(e1("direct"), None, None, "request_mismatch"),
                       compute_metrics(None, None, None, "request_mismatch")])
    assert cells["s_solution_present"] == (1.0, 1)
    assert cells["fully_pragmatic"] == (None, 0)


def test_emphasis_neither_is_not_s_solution():
    m = compute_metrics(e1("direct", "alternate"), e2("intended_goal"), e3("neither"), "request_mismatch")
    assert m["emphasis"] == "neither" and m["fully_pragmatic"] is False and m["fully_cooperative"] is False
    cells = aggregate([m])
    assert cells["emphasis_s_solution"] == (0.0, 1) and cells["emphasis_neither"] == (1.0, 1)
    m = compute_metrics(None, None, e3("Approach_2 "), "request_mismatch")
    assert m["emphasis"] == "i_solution"
    assert compute_metrics(None, None, e3("something else"), "request_mismatch")["emphasis"] is None


def test_labels_are_normalized():
    m = compute_metrics(e1("Direct", " alternate"), e2("Intended_Goal"), None, "request_mismatch")
    assert m["s_solution_present"] is True and m["i_solution_present"] is True and m["misconception_identified"] is True
