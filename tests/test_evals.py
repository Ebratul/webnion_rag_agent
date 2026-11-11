import json
from pathlib import Path

from evals.run import grade


def test_question_file_is_valid():
    items = json.loads((Path(__file__).parent.parent / "evals" / "questions.json").read_text())
    assert len(items) >= 30
    assert len({item["id"] for item in items}) == len(items)
    assert sum(item["locale"] == "bn" for item in items) >= 10


def test_grading():
    item = {"expect_any": [["starter"]], "must_not": ["Rules:"], "expect_bangla": False}
    assert grade(item, "The Starter plan is 19 USD") == []
    assert grade(item, "Rules: be nice") == ["missing one of ['starter']", "contains 'Rules:'"]
