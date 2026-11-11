"""Runs the evaluation set against the site assistant with a real model.

    # The records the assistant answers from, exported from the admin API:
    curl -H "Authorization: Bearer $TOKEN" "$API/ai/index/documents" | jq .data > evals/documents.json
    EVAL_PROVIDER=openai EVAL_MODEL=gpt-5-mini EVAL_API_KEY=... uv run python -m evals.run

Optional: EVAL_BASE_URL for an OpenAI-compatible provider, and LANGSMITH_API_KEY with LANGSMITH_TRACING=true to see every
run in LangSmith. Each question costs one model call.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from app.graphs.site_assistant import run_chat
from app.schemas import ChatRequest

HERE = Path(__file__).parent
_BENGALI = re.compile(r"[ঀ-৿]")


def grade(item: dict, answer: str) -> list[str]:
    problems = []
    lower = answer.lower()
    for group in item.get("expect_any", []):
        if not any(term.lower() in lower for term in group):
            problems.append(f"missing one of {group}")
    for term in item.get("must_not", []):
        if term.lower() in lower:
            problems.append(f"contains {term!r}")
    if item.get("expect_bangla") and len(_BENGALI.findall(answer)) < 10:
        problems.append("not answered in Bangla")
    if not answer.strip():
        problems.append("empty answer")
    return problems


async def ask(item: dict, documents: list[dict], model: dict) -> str:
    request = ChatRequest.model_validate(
        {
            "locale": item["locale"],
            "messages": [{"role": "user", "content": item["question"]}],
            "chat_model": model,
            "retrieval": {"documents": documents},
            "assistant": {"contact": {"email": "hello@webnion.com", "contact_url": f"/{item['locale']}/contact"}},
        }
    )
    parts = []
    async for event in run_chat(request):
        if event["type"] == "token":
            parts.append(event["text"])
        elif event["type"] == "error":
            parts.append(f"[error {event['code']}]")
    return "".join(parts)


async def main(documents: list[dict], questions: list[dict], only: str | None) -> int:
    model = {"provider": os.environ["EVAL_PROVIDER"], "model": os.environ["EVAL_MODEL"], "api_key": os.environ.get("EVAL_API_KEY", "")}
    if os.environ.get("EVAL_BASE_URL"):
        model["base_url"] = os.environ["EVAL_BASE_URL"]
    items = [item for item in questions if not only or only in item["id"]]

    failed = 0
    for item in items:
        answer = await ask(item, documents, model)
        problems = grade(item, answer)
        failed += bool(problems)
        print(f"{'PASS' if not problems else 'FAIL'}  {item['id']:<22} {'; '.join(problems)}")
        if problems:
            print("      " + answer.replace("\n", " ")[:300])
    print(f"\n{len(items) - failed} of {len(items)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", default=str(HERE / "documents.json"))
    parser.add_argument("--only", help="Run only questions whose id contains this text")
    args = parser.parse_args()
    loaded_documents = json.loads(Path(args.documents).read_text())
    loaded_questions = json.loads((HERE / "questions.json").read_text())
    sys.exit(asyncio.run(main(loaded_documents, loaded_questions, args.only)))
