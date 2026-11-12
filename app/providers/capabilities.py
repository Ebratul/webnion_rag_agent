"""The checks behind the Test button in AI Settings."""

import time

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from app.providers.chat import build_chat_model
from app.providers.embeddings import build_embeddings
from app.providers.errors import ProviderError, classify
from app.providers.images import generate_picture
from app.schemas import ModelTestRequest, ModelTestResult

# Hard errors stop the test. Anything else in the optional checks only means that feature is missing.
_FATAL = {"auth", "model", "rate_limit", "unavailable", "unsafe_url", "not_supported", "loading"}


@tool
def get_office_hours(city: str) -> str:
    """Returns the office hours for a city."""
    return f"{city}: 9am to 6pm"


class _Probe(BaseModel):
    word: str
    number: int


async def _chat(request: ModelTestRequest, result: ModelTestResult) -> None:
    # Reasoning models spend part of the allowance thinking, so the limit is generous for a one-word answer
    model = await build_chat_model(request.model, max_tokens=1024, timeout=60)
    await model.ainvoke([HumanMessage("Reply with the single word OK.")])
    result.answers = True

    try:
        reply = await model.bind_tools([get_office_hours]).ainvoke([HumanMessage("What are the office hours in Dhaka? Use the get_office_hours tool.")])
        result.tools = bool(getattr(reply, "tool_calls", None))
    except Exception as error:
        if classify(error).code in _FATAL - {"not_supported"}:
            raise
        result.tools = False

    try:
        probe = await model.with_structured_output(_Probe).ainvoke([HumanMessage("Return word 'blue' and number 7.")])
        result.structured = isinstance(probe, _Probe) and probe.number == 7
    except Exception as error:
        if classify(error).code in _FATAL - {"not_supported"}:
            raise
        result.structured = False


async def run_test(request: ModelTestRequest) -> ModelTestResult:
    result = ModelTestResult(ok=False)
    started = time.perf_counter()
    try:
        if request.kind == "chat":
            await _chat(request, result)
        elif request.kind == "embedding":
            embeddings = await build_embeddings(request.model)
            vector = await embeddings.aembed_query("Webnion builds websites and apps.")
            result.answers = True
            result.dimension = len(vector)
        else:
            if request.model.provider not in ("openai", "google_genai", "huggingface", "cloudflare", "fake"):
                raise ProviderError("not_supported")
            result.answers = True
            if request.make_picture:
                await generate_picture(request.model, "A calm abstract gradient of soft purple and orange shapes, no text", "square")
                result.picture = True
        result.ok = True
    except Exception as error:  # noqa: BLE001 - every failure becomes a plain message for the admin
        result.ok = False
        result.error = classify(error, request.model.api_key.get_secret_value()).as_dict()
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result
