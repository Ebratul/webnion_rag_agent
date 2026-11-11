"""The site assistant (PLAN.md section 5.1): guard, retrieve, answer.

The assistant's only action is to propose an inquiry. It never sends one: the proposal goes to the visitor's browser,
the visitor checks it, and the browser sends it through the normal inquiry form endpoint after the visitor presses Send.
"""

import re
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.graphs.prompts import BLOCKED_REPLY, INQUIRY_REPLY, REFUSAL_REPLY, site_assistant_system
from app.providers.chat import build_chat_model
from app.providers.errors import classify
from app.rag.documents import Passage, detect_language
from app.rag.store import retrieve
from app.schemas import ChatRequest

_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,}$")
_PHONE = re.compile(r"^\+?[0-9][0-9\s()-]{6,18}$")


class InquiryProposal(BaseModel):
    """What the visitor shared, shown back to them before anything is sent."""

    name: str = Field(default="", max_length=100)
    email: str = Field(default="", max_length=200)
    phone: str = Field(default="", max_length=30)
    service: str = Field(default="", max_length=150)
    message: str = Field(default="", max_length=2000)

    @field_validator("*", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return str(value or "").strip()[:2000]

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return value.lower() if _EMAIL.match(value) else ""

    @field_validator("phone")
    @classmethod
    def _phone(cls, value: str) -> str:
        return value if _PHONE.match(value) else ""


@tool
def propose_inquiry(name: str, email: str = "", phone: str = "", service: str = "", message: str = "") -> str:
    """Shows the visitor a summary of their project inquiry to check and send. Call it once you have an email or phone
    number and a description of what they need.

    Args:
        name: The visitor's name.
        email: The visitor's email address, if given.
        phone: The visitor's phone number, if given.
        service: The service they are interested in, if known.
        message: A short description of what they need, in their words.
    """
    return "Shown to the visitor for confirmation."


class State(TypedDict, total=False):
    request: ChatRequest
    question: str
    language: str
    blocked: str
    passages: list[Passage]


def guard(state: State) -> State:
    request = state["request"]
    user_turns = [turn.content.strip() for turn in request.messages if turn.role == "user"]
    question = user_turns[-1]
    language = detect_language(question, request.locale)
    if not question:
        return {"question": question, "language": language, "blocked": "empty"}
    # The same message three times in a row is a loop or a bot
    if len(user_turns) >= 3 and len({turn.lower() for turn in user_turns[-3:]}) == 1:
        return {"question": question, "language": language, "blocked": "repeated"}
    return {"question": question, "language": language}


async def retrieve_passages(state: State) -> State:
    request = state["request"]
    user_turns = [turn.content for turn in request.messages if turn.role == "user"]
    # A follow-up like "and the price?" needs the question before it to find the right passages
    query = " ".join(user_turns[-2:])[-1200:]
    try:
        return {"passages": await retrieve(request.retrieval, query, state["language"])}
    except Exception as error:  # noqa: BLE001 - the assistant still answers from the contact details
        get_stream_writer()({"type": "warning", "code": classify(error).code, "where": "search"})
        return {"passages": []}


def blocked_reply(state: State) -> State:
    get_stream_writer()({"type": "token", "text": BLOCKED_REPLY[state["blocked"]][state["language"]]})
    return {}


def _history(request: ChatRequest) -> list[BaseMessage]:
    return [HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content) for turn in request.messages[-10:]]


async def answer(state: State) -> State:
    request = state["request"]
    writer = get_stream_writer()
    language = state["language"]
    passages = state.get("passages", [])
    can_take_inquiry = request.assistant.allow_inquiry and request.chat_model.tools

    model = await build_chat_model(request.chat_model, max_tokens=request.assistant.max_output_tokens, timeout=60)
    runnable = model.bind_tools([propose_inquiry]) if can_take_inquiry else model
    messages = [SystemMessage(site_assistant_system(request.assistant, language, passages, can_take_inquiry, request.audience)), *_history(request)]

    full: AIMessageChunk | None = None
    wrote = False
    async for chunk in runnable.astream(messages):
        full = chunk if full is None else full + chunk
        text = chunk.text if isinstance(chunk.text, str) else ""
        if text:
            wrote = True
            writer({"type": "token", "text": text})

    if full is None:
        full = AIMessageChunk(content="")
    stop = (full.response_metadata or {}).get("stop_reason") or (full.response_metadata or {}).get("finish_reason")

    if stop == "refusal" and not wrote:
        writer({"type": "token", "text": REFUSAL_REPLY[language]})
        wrote = True

    calls = [call for call in (full.tool_calls or []) if call.get("name") == "propose_inquiry"]
    if calls and can_take_inquiry:
        try:
            proposal = InquiryProposal.model_validate(calls[0].get("args") or {})
        except ValidationError:
            proposal = None
        if proposal and (proposal.email or proposal.phone):
            if not wrote:
                writer({"type": "token", "text": INQUIRY_REPLY[language]})
                wrote = True
            writer({"type": "inquiry", "proposal": proposal.model_dump()})

    if not wrote:
        writer({"type": "token", "text": REFUSAL_REPLY[language]})

    used = [passage for passage in passages if passage.url][:3]
    seen: set[str] = set()
    sources = []
    for passage in used:
        if passage.url not in seen:
            seen.add(passage.url)
            sources.append({"title": passage.title, "url": passage.url})
    if sources:
        writer({"type": "sources", "items": sources})

    usage = full.usage_metadata or {}
    writer({"type": "usage", "input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))})
    return {}


def _after_guard(state: State) -> str:
    return "blocked" if state.get("blocked") else "retrieve"


def build_graph() -> Any:
    graph = StateGraph(State)
    graph.add_node("guard", guard)
    graph.add_node("retrieve", retrieve_passages)
    graph.add_node("answer", answer)
    graph.add_node("blocked", blocked_reply)
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", _after_guard, {"blocked": "blocked", "retrieve": "retrieve"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("blocked", END)
    return graph.compile()


site_assistant = build_graph()


async def run_chat(request: ChatRequest):
    """Yields the events the Express API streams to the browser: token, inquiry, sources, usage, error."""
    try:
        async for event in site_assistant.astream({"request": request}, stream_mode="custom"):
            yield event
    except Exception as error:  # noqa: BLE001 - every failure becomes an error event
        yield {"type": "error", **classify(error, request.chat_model.api_key.get_secret_value()).as_dict()}
