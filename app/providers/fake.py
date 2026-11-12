"""A stand-in model for local testing. It calls no AI service and costs nothing.

It is used only when AGENT_ALLOW_FAKE_PROVIDER=1, which is never set on Vercel.
"""

import hashlib
import json
import math
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from app.config import get_settings
from app.providers.errors import ProviderError

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SCHEMA = re.compile(r"JSON schema:\s*(\{.*\})\s*$", re.S)


def require_fake_allowed() -> None:
    if not get_settings().agent_allow_fake_provider:
        raise ProviderError("not_supported", "the fake provider is only for local testing")


def _example(schema: dict[str, Any], defs: dict[str, Any], name: str = "") -> Any:
    """Builds a small valid value for a JSON schema, so the JSON fallback path can be tested."""
    if "$ref" in schema:
        return _example(defs[schema["$ref"].split("/")[-1]], defs, name)
    if "anyOf" in schema:
        return _example(next(option for option in schema["anyOf"] if option.get("type") != "null"), defs, name)
    kind = schema.get("type")
    if kind == "object":
        return {key: _example(value, defs, key) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        count = max(schema.get("minItems", 0), min(3, schema.get("maxItems", 3)))
        return [_example(schema.get("items", {}), defs, name) for _ in range(count)]
    if kind == "integer":
        return schema.get("minimum", 0)
    if kind == "boolean":
        return True
    text = f"Sample {name.replace('_', ' ') or 'text'} for a growing business website"
    return text[: schema.get("maxLength", 200)]


class FakeChatModel(BaseChatModel):
    tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "webnion-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        names = [getattr(tool, "name", None) or getattr(tool, "__name__", "tool") for tool in tools]
        return self.model_copy(update={"tool_names": names})

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        system = "\n".join(str(message.content) for message in messages if isinstance(message, SystemMessage))
        last = next((str(message.content) for message in reversed(messages) if isinstance(message, HumanMessage)), "")
        prompt = system + "\n" + last
        usage = {"input_tokens": len(prompt) // 4, "output_tokens": 40, "total_tokens": len(prompt) // 4 + 40}

        schema = _SCHEMA.search(last)
        if schema:
            parsed = json.loads(schema.group(1))
            return AIMessage(content=json.dumps(_example(parsed, parsed.get("$defs", {}))), usage_metadata=usage)

        named = next((name for name in self.tool_names if name in last and name != "propose_inquiry"), None)
        if named:
            return AIMessage(content="", tool_calls=[{"name": named, "args": {"city": "Dhaka"}, "id": "fake-call-0"}], usage_metadata=usage)

        if self.tool_names and _EMAIL.search(last):
            email = _EMAIL.search(last).group(0)
            args = {"name": "Test Visitor", "email": email, "phone": "", "service": "", "message": last[:500]}
            return AIMessage(content="", tool_calls=[{"name": self.tool_names[0], "args": args, "id": "fake-call-1"}], usage_metadata=usage)

        if "## " in last and "Edit" in system:
            return AIMessage(content=last.split("<article>")[-1].split("</article>")[0].strip(), usage_metadata=usage)

        passages = re.findall(r"<passage[^>]*>\s*(.*?)\s*</passage>", system, re.S)
        if passages:
            text = "From the site: " + passages[0][:280].replace("\n", " ")
        else:
            text = "This is a test answer from the local fake model.\n\nIt has a second paragraph.\n\n- One point\n- Another point"
        return AIMessage(content=text, usage_metadata=usage)

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._reply(messages)
        if reply.tool_calls:
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": 0} for call in reply.tool_calls],
                usage_metadata=reply.usage_metadata,
            )
            yield ChatGenerationChunk(message=chunk)
            return
        words = str(reply.content).split(" ")
        for position, word in enumerate(words):
            last = position == len(words) - 1
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + ("" if last else " "), usage_metadata=reply.usage_metadata if last else None))


class HashingEmbeddings(Embeddings):
    """Local stand-in for an embedding model: words hashed into buckets, so texts sharing words come out close."""

    def __init__(self, size: int = 384) -> None:
        self.size = size

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in re.findall(r"[\w\u0980-\u09FF]+", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big")
            vector[bucket % self.size] += 1.0 if bucket & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        # An empty text still gets a valid vector
        return [value / norm for value in vector] if any(vector) else [1.0 / math.sqrt(self.size)] * self.size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
