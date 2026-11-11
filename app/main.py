"""The FastAPI app. Every route except /health needs a request signed by the Express API."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.graphs.blog_writer import make_outline, run_translate, run_write
from app.graphs.site_assistant import run_chat
from app.providers.capabilities import run_test
from app.providers.errors import classify
from app.providers.images import generate_picture
from app.rag.store import PineconeStore, retrieve
from app.schemas import (
    ChatRequest,
    ImageRequest,
    IndexDeleteRequest,
    IndexUpsertRequest,
    ModelTestRequest,
    OutlineRequest,
    SearchRequest,
    TranslateRequest,
    WriteRequest,
)
from app.security import verify_signature
from app.tracing import traced

logger = logging.getLogger("webnion.agent")

app = FastAPI(title="Webnion AI service", docs_url=None, redoc_url=None, openapi_url=None)
signed = [Depends(verify_signature)]


@app.exception_handler(RequestValidationError)
async def _invalid(_request: Request, error: RequestValidationError) -> JSONResponse:
    # Only where and what, never the submitted value: it may be an API key
    problems = [
        {"field": ".".join(str(part) for part in issue.get("loc", ())[1:]), "message": str(issue.get("msg", ""))[:200]} for issue in error.errors()[:10]
    ]
    return JSONResponse(status_code=422, content={"code": "invalid", "message": "The request is not valid", "problems": problems})


@app.exception_handler(StarletteHTTPException)
async def _http(_request: Request, error: StarletteHTTPException) -> JSONResponse:
    detail = error.detail if isinstance(error.detail, dict) else {"code": "error", "message": str(error.detail)}
    return JSONResponse(status_code=error.status_code, content=detail)


@app.exception_handler(Exception)
async def _unexpected(_request: Request, error: Exception) -> JSONResponse:
    logger.error("Unexpected error: %s", type(error).__name__)
    return JSONResponse(status_code=500, content={"code": "error", "message": "Something went wrong in the AI service"})


@app.middleware("http")
async def _headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _sse(events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        async for event in events:
            yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
        yield b"event: done\ndata: {}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "configured": get_settings().configured}


@app.post("/v1/models/test", dependencies=signed)
async def test_model(request: ModelTestRequest) -> dict[str, Any]:
    with traced(request.langsmith, "model-test", tags=[request.kind, request.model.provider]):
        return (await run_test(request)).model_dump()


@app.post("/v1/chat", dependencies=signed)
async def chat(request: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[dict[str, Any]]:
        with traced(
            request.langsmith,
            "site-assistant",
            tags=[request.audience, request.chat_model.provider],
            metadata={"locale": request.locale, "model": request.chat_model.model},
        ):
            async for event in run_chat(request):
                yield event

    return _sse(events())


@app.post("/v1/blog/outline", dependencies=signed)
async def blog_outline(request: OutlineRequest) -> JSONResponse:
    try:
        with traced(request.langsmith, "blog-outline", tags=[request.writing_model.provider]):
            outline = await make_outline(request)
        return JSONResponse({"outline": outline.model_dump()})
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=502, content=classify(error, request.writing_model.api_key.get_secret_value()).as_dict())


@app.post("/v1/blog/write", dependencies=signed)
async def blog_write(request: WriteRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[dict[str, Any]]:
        with traced(request.langsmith, "blog-write", tags=[request.writing_model.provider]):
            async for event in run_write(request):
                yield event

    return _sse(events())


@app.post("/v1/blog/translate", dependencies=signed)
async def blog_translate(request: TranslateRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[dict[str, Any]]:
        with traced(request.langsmith, "blog-translate", tags=[request.writing_model.provider]):
            async for event in run_translate(request):
                yield event

    return _sse(events())


@app.post("/v1/images", dependencies=signed)
async def images(request: ImageRequest) -> JSONResponse:
    try:
        with traced(request.langsmith, "picture", tags=[request.image_model.provider]):
            picture = await generate_picture(request.image_model, request.prompt, request.aspect)
        return JSONResponse(picture.as_dict())
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=502, content=classify(error, request.image_model.api_key.get_secret_value()).as_dict())


async def _store(request: IndexUpsertRequest) -> PineconeStore:
    from app.providers.embeddings import build_embeddings

    return PineconeStore(request.pinecone, await build_embeddings(request.embedding))


@app.post("/v1/index/upsert", dependencies=signed)
async def index_upsert(request: IndexUpsertRequest) -> JSONResponse:
    try:
        result = await (await _store(request)).upsert(request.documents, replace_all=request.replace_all)
        return JSONResponse(result)
    except Exception as error:  # noqa: BLE001
        secrets = (request.embedding.api_key.get_secret_value(), request.pinecone.api_key.get_secret_value())
        return JSONResponse(status_code=502, content=classify(error, *secrets).as_dict())


@app.post("/v1/index/delete", dependencies=signed)
async def index_delete(request: IndexDeleteRequest) -> JSONResponse:
    try:
        from app.providers.fake import HashingEmbeddings

        # Deleting needs no embeddings, only the ids
        store = PineconeStore(request.pinecone, HashingEmbeddings(8))
        return JSONResponse({"removed": await store.delete(request.items)})
    except Exception as error:  # noqa: BLE001
        return JSONResponse(status_code=502, content=classify(error, request.pinecone.api_key.get_secret_value()).as_dict())


@app.post("/v1/search", dependencies=signed)
async def search(request: SearchRequest) -> JSONResponse:
    try:
        passages = await retrieve(request.retrieval, request.query, request.locale)
        return JSONResponse(
            {
                "results": [
                    {
                        "title": passage.title,
                        "url": passage.url,
                        "type": passage.type,
                        "locale": passage.locale,
                        "score": round(passage.score, 4),
                        "text": passage.text[:400],
                    }
                    for passage in passages
                ]
            }
        )
    except Exception as error:  # noqa: BLE001
        secrets = [config.api_key.get_secret_value() for config in (request.retrieval.embedding, request.retrieval.pinecone) if config]
        return JSONResponse(status_code=502, content=classify(error, *secrets).as_dict())
