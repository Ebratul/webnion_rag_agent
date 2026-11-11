"""The blog writer (PLAN.md section 5.2).

Step 1 makes an outline. The admin edits and approves it. Step 2 writes the article, edits it, writes the SEO fields and
plans the pictures. Pictures and the Bangla translation are separate requests, so no single request runs into Vercel's
five-minute limit.
"""

import asyncio
import json
import re
import time
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from app.graphs.prompts import (
    EDIT_SYSTEM,
    LENGTHS,
    TRANSLATE_SYSTEM,
    json_instruction,
    outline_prompt,
    pictures_prompt,
    section_prompt,
    writer_system,
)
from app.providers.chat import build_chat_model, text_of
from app.providers.errors import ProviderError, classify
from app.rag.documents import Passage
from app.rag.store import retrieve
from app.schemas import ChatModelConfig, Outline, OutlineRequest, PostDraft, TranslateRequest, WriteRequest

# Errors that mean the model cannot be used at all, as opposed to one feature being missing
FATAL = {"auth", "model", "rate_limit", "unavailable", "unsafe_url", "not_supported", "loading", "content_filter"}
# Leaves room inside Vercel's five minutes for saving the draft
SKIP_EDIT_AFTER = 170
SKIP_PICTURES_AFTER = 230
PARALLEL_SECTIONS = 3


class SeoFields(BaseModel):
    seo_title: str
    meta_description: str
    excerpt: str
    tags: list[str]


class CoverPicture(BaseModel):
    prompt: str
    alt: str


class SectionPicture(BaseModel):
    section: int
    prompt: str
    alt: str


class PicturePlan(BaseModel):
    cover: CoverPicture
    sections: list[SectionPicture]


class TranslatedMeta(BaseModel):
    title: str
    excerpt: str
    seo_title: str
    meta_description: str
    tags: list[str]


def _json_from(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    return text[start : end + 1]


async def structured[T: BaseModel](model: BaseChatModel, schema: type[T], messages: list[BaseMessage], native: bool) -> T:
    """Structured output when the model supports it, otherwise JSON asked for in the prompt, checked and retried once."""
    if native:
        try:
            result = await model.with_structured_output(schema).ainvoke(messages)
            return result if isinstance(result, schema) else schema.model_validate(result)
        except Exception as error:
            if classify(error).code in FATAL:
                raise

    last = messages[-1]
    asked = [*messages[:-1], HumanMessage(str(last.content) + json_instruction(schema))]
    reply = await model.ainvoke(asked)
    try:
        return schema.model_validate_json(_json_from(text_of(reply)))
    except (ValueError, ValidationError) as error:
        retry = [*asked, reply, HumanMessage(f"That reply was not valid: {str(error)[:300]}. Reply with only the corrected JSON object.")]
        second = await model.ainvoke(retry)
        try:
            return schema.model_validate_json(_json_from(text_of(second)))
        except (ValueError, ValidationError) as final:
            raise ProviderError("bad_request", "the model did not return the expected format") from final


# The article format: "## " headings, paragraphs and "- " lists, turned into the site's content blocks

_INLINE = re.compile(r"(\*\*|__|`)")
_TAG = re.compile(r"<[^>]{1,200}>")
_LIST = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")


def _clean(text: str) -> str:
    return _TAG.sub("", _INLINE.sub("", text)).strip()[:5000]


def parse_article(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if paragraph:
            joined = _clean(" ".join(paragraph))
            if joined:
                blocks.append({"type": "paragraph", "text": joined})
            paragraph.clear()
        if items:
            blocks.append({"type": "list", "items": items[:50]})
            items.clear()

    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if line.startswith("```"):
            continue
        if not line:
            flush()
        elif line.startswith("#"):
            flush()
            heading = _clean(line.lstrip("#"))
            if heading:
                blocks.append({"type": "heading", "text": heading[:200]})
        elif _LIST.match(line):
            if paragraph:
                flush()
            item = _clean(_LIST.sub("", line))
            if item:
                items.append(item)
        else:
            if items:
                flush()
            paragraph.append(line)
    flush()
    return blocks


def to_markdown(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block["type"] == "heading":
            parts.append(f"## {block['text']}")
        elif block["type"] == "paragraph":
            parts.append(block["text"])
        elif block["type"] == "list":
            parts.append("\n".join(f"- {item}" for item in block["items"]))
    return "\n\n".join(parts)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80].rstrip("-") or "post"


async def _passages(request: OutlineRequest) -> list[Passage]:
    query = " ".join([request.brief.topic, *request.brief.keywords])
    try:
        return await retrieve(request.retrieval, query, "en", top_k=6)
    except Exception as error:
        if classify(error).code == "auth":
            raise
        return []


async def make_outline(request: OutlineRequest) -> Outline:
    passages = await _passages(request)
    model = await build_chat_model(request.writing_model, max_tokens=4096, timeout=120)
    messages: list[BaseMessage] = [SystemMessage(writer_system(request.site_name, passages)), HumanMessage(outline_prompt(request.brief))]
    return await structured(model, Outline, messages, request.writing_model.structured)


class WriteState(TypedDict, total=False):
    request: WriteRequest
    started: float
    passages: list[Passage]
    sections: list[str]
    article: str
    seo: SeoFields
    pictures: PicturePlan | None


async def write_retrieve(state: WriteState) -> WriteState:
    get_stream_writer()({"type": "progress", "step": "research", "message": "Reading the site's content"})
    return {"passages": await _passages(state["request"])}


async def write_sections(state: WriteState) -> WriteState:
    request = state["request"]
    writer = get_stream_writer()
    outline = request.outline
    headings = [section.heading for section in outline.sections]
    words = max(120, LENGTHS[request.brief.length]["words"] // len(headings))
    model = await build_chat_model(request.writing_model, max_tokens=4096, timeout=150)
    system = SystemMessage(writer_system(request.site_name, state.get("passages", [])))
    limit = asyncio.Semaphore(PARALLEL_SECTIONS)
    done = 0

    async def write(index: int) -> str:
        nonlocal done
        async with limit:
            prompt = section_prompt(outline.title, outline.angle, headings, index, outline.sections[index].points, words, request.brief)
            reply = await model.ainvoke([system, HumanMessage(prompt)])
            done += 1
            writer({"type": "progress", "step": "write", "message": f"Wrote section {done} of {len(headings)}"})
            body = "\n".join(line for line in text_of(reply).splitlines() if not line.lstrip().startswith("#"))
            return f"## {headings[index]}\n\n{body.strip()}"

    writer({"type": "progress", "step": "write", "message": f"Writing {len(headings)} sections"})
    sections = await asyncio.gather(*(write(index) for index in range(len(headings))))
    return {"sections": list(sections), "article": "\n\n".join(sections)}


async def edit(state: WriteState) -> WriteState:
    request = state["request"]
    writer = get_stream_writer()
    article = state["article"]
    if time.monotonic() - state["started"] > SKIP_EDIT_AFTER:
        writer({"type": "progress", "step": "edit", "message": "Skipped the edit pass to stay within the time limit"})
        return {}
    writer({"type": "progress", "step": "edit", "message": "Editing the draft"})
    try:
        model = await build_chat_model(request.writing_model, max_tokens=8192, timeout=150)
        reply = await model.ainvoke([SystemMessage(EDIT_SYSTEM), HumanMessage(f"<article>\n{article}\n</article>")])
        edited = text_of(reply).replace("<article>", "").replace("</article>", "").strip()
    except Exception as error:
        if classify(error).code in ("auth", "model"):
            raise
        return {}
    # An edit that lost headings or most of the text is worse than no edit
    if edited.count("## ") < article.count("## ") or len(edited) < 0.6 * len(article):
        return {}
    return {"article": edited}


async def seo(state: WriteState) -> WriteState:
    request = state["request"]
    get_stream_writer()({"type": "progress", "step": "seo", "message": "Writing the title, description and tags"})
    model = await build_chat_model(request.writing_model, max_tokens=2048, timeout=90)
    prompt = (
        f"Article title: {request.outline.title}\n\n<article>\n{state['article'][:12000]}\n</article>\n\n"
        "Write an SEO title of at most 60 characters, a meta description of at most 155 characters, a one or two sentence "
        "excerpt for the blog list, and 3 to 6 short lowercase tags."
    )
    fields = await structured(model, SeoFields, [SystemMessage("You are an SEO editor."), HumanMessage(prompt)], request.writing_model.structured)
    return {"seo": fields}


async def plan_pictures(state: WriteState) -> WriteState:
    request = state["request"]
    if request.max_pictures == 0 or time.monotonic() - state["started"] > SKIP_PICTURES_AFTER:
        return {"pictures": None}
    get_stream_writer()({"type": "progress", "step": "pictures", "message": "Planning the pictures"})
    headings = [section.heading for section in request.outline.sections]
    model = await build_chat_model(request.writing_model, max_tokens=2048, timeout=90)
    try:
        plan = await structured(
            model,
            PicturePlan,
            [HumanMessage(pictures_prompt(request.outline.title, headings, request.max_pictures, request.picture_style))],
            request.writing_model.structured,
        )
    except ProviderError as error:
        if error.code in ("auth", "model"):
            raise
        return {"pictures": None}
    return {"pictures": plan}


def _build_write_graph() -> Any:
    graph = StateGraph(WriteState)
    graph.add_node("research", write_retrieve)
    graph.add_node("write_sections", write_sections)
    graph.add_node("edit", edit)
    graph.add_node("seo", seo)
    graph.add_node("picture_prompts", plan_pictures)
    graph.add_edge(START, "research")
    graph.add_edge("research", "write_sections")
    graph.add_edge("write_sections", "edit")
    graph.add_edge("edit", "seo")
    graph.add_edge("seo", "picture_prompts")
    graph.add_edge("picture_prompts", END)
    return graph.compile()


blog_writer = _build_write_graph()


def _pictures_out(plan: PicturePlan | None, sections: int, limit: int) -> list[dict[str, Any]]:
    if plan is None or limit == 0:
        return []
    pictures = [{"slot": "cover", "prompt": plan.cover.prompt[:1500], "alt": plan.cover.alt[:200]}]
    used: set[int] = set()
    for picture in plan.sections:
        if len(pictures) >= limit:
            break
        if 0 <= picture.section < sections and picture.section not in used:
            used.add(picture.section)
            pictures.append({"slot": picture.section, "prompt": picture.prompt[:1500], "alt": picture.alt[:200]})
    return pictures


async def run_write(request: WriteRequest):
    """Yields progress events, then one result event with the English draft and the picture plan."""
    try:
        final: dict[str, Any] = {}
        async for mode, event in blog_writer.astream({"request": request, "started": time.monotonic()}, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield event
            else:
                final = event
        seo_fields: SeoFields = final["seo"]
        post = {
            "title": request.outline.title[:200],
            "slug": slugify(request.outline.title),
            "excerpt": seo_fields.excerpt.strip()[:300],
            "blocks": parse_article(final["article"]),
            "seo_title": seo_fields.seo_title.strip()[:70],
            "seo_description": seo_fields.meta_description.strip()[:160],
            "tags": [re.sub(r"\s+", " ", tag.strip().lower())[:40] for tag in seo_fields.tags if tag.strip()][:6],
        }
        yield {"type": "result", "post": post, "pictures": _pictures_out(final.get("pictures"), len(request.outline.sections), request.max_pictures)}
    except Exception as error:  # noqa: BLE001
        yield {"type": "error", **classify(error, request.writing_model.api_key.get_secret_value()).as_dict()}


# Translation into Bangla, one section at a time so each reply stays short


def _sections(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for block in blocks:
        if block["type"] == "heading" or not groups:
            groups.append([])
        groups[-1].append(block)
    return groups


async def run_translate(request: TranslateRequest):
    config: ChatModelConfig = request.writing_model
    try:
        model = await build_chat_model(config, max_tokens=8192, timeout=150)
        blocks = [block.model_dump() for block in request.post.blocks]
        groups = _sections(blocks)
        limit = asyncio.Semaphore(PARALLEL_SECTIONS)
        done = 0
        yield {"type": "progress", "step": "translate", "message": f"Translating {len(groups)} sections into Bangla"}

        async def translate(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal done
            text_blocks = [block for block in group if block["type"] != "image"]
            images = [block for block in group if block["type"] == "image"]
            if not text_blocks:
                return images
            async with limit:
                reply = await model.ainvoke([SystemMessage(TRANSLATE_SYSTEM), HumanMessage(to_markdown(text_blocks))])
            done += 1
            translated = parse_article(text_of(reply)) or text_blocks
            # Pictures stay where they were: after the section heading, or first when the section has none
            if images:
                position = 1 if translated and translated[0]["type"] == "heading" else 0
                translated[position:position] = images
            return translated

        tasks = [asyncio.create_task(translate(group)) for group in groups]
        for finished in asyncio.as_completed(tasks):
            await finished
            yield {"type": "progress", "step": "translate", "message": f"Translated {done} of {len(groups)} sections"}
        translated_blocks = [block for task in tasks for block in task.result()]

        meta_source = json.dumps(
            {
                "title": request.post.title,
                "excerpt": request.post.excerpt,
                "seo_title": request.post.seo_title,
                "meta_description": request.post.seo_description,
                "tags": request.post.tags,
            },
            ensure_ascii=False,
        )
        meta = await structured(
            model,
            TranslatedMeta,
            [SystemMessage(TRANSLATE_SYSTEM), HumanMessage(f"Translate every value in this JSON into Bangla:\n{meta_source}")],
            config.structured,
        )
        post = {
            "title": meta.title.strip()[:200],
            "excerpt": meta.excerpt.strip()[:300],
            "blocks": translated_blocks,
            "seo_title": meta.seo_title.strip()[:70],
            "seo_description": meta.meta_description.strip()[:160],
            "tags": [tag.strip()[:40] for tag in meta.tags if tag.strip()][:6],
        }
        yield {"type": "result", "post": post}
    except Exception as error:  # noqa: BLE001
        yield {"type": "error", **classify(error, config.api_key.get_secret_value()).as_dict()}


def outline_to_dict(outline: Outline) -> dict[str, Any]:
    return outline.model_dump()


__all__ = ["PostDraft", "make_outline", "outline_to_dict", "parse_article", "run_translate", "run_write"]
