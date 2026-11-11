"""The Pinecone search index: create, update, delete and search."""

import asyncio
from typing import Any

from langchain_core.embeddings import Embeddings

from app.providers.embeddings import build_embeddings
from app.providers.errors import ProviderError
from app.rag.documents import Passage, id_prefix, keyword_search, to_passages
from app.schemas import Document, IndexItem, PineconeConfig, Retrieval

NAMESPACES = ("en", "bn")
EMBED_BATCH = 64
UPSERT_BATCH = 100
# Passages below this similarity are too far from the question to help
MIN_SCORE = 0.2


def _pinecone_client(config: PineconeConfig) -> Any:
    from pinecone import Pinecone

    return Pinecone(api_key=config.api_key.get_secret_value())


class PineconeStore:
    """Wraps the synchronous Pinecone client. Every call runs in a thread so the event loop stays free."""

    def __init__(self, config: PineconeConfig, embeddings: Embeddings, client: Any | None = None) -> None:
        self.config = config
        self.embeddings = embeddings
        self.client = client or _pinecone_client(config)
        self._index: Any | None = None

    async def _call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(function, *args, **kwargs)

    async def index(self, dimension: int | None = None) -> Any:
        """Returns the index, creating it when it does not exist and a dimension is known."""
        if self._index is not None:
            return self._index
        name = self.config.index
        if await self._call(self.client.has_index, name):
            described = await self._call(self.client.describe_index, name)
            existing = getattr(described, "dimension", None)
            if dimension and existing and existing != dimension:
                raise ProviderError(
                    "bad_request",
                    f"the index '{name}' holds vectors of size {existing}, but the embedding model makes size {dimension}. "
                    "Use a new index name or the earlier embedding model",
                )
        elif dimension:
            from pinecone import ServerlessSpec

            await self._call(
                self.client.create_index,
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.config.cloud, region=self.config.region),
            )
            # A new index takes a few seconds before it accepts writes
            for _ in range(60):
                described = await self._call(self.client.describe_index, name)
                status = getattr(described, "status", None) or {}
                ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
                if ready:
                    break
                await asyncio.sleep(1)
        else:
            raise ProviderError("bad_request", f"the index '{name}' does not exist yet. Run Rebuild in AI Settings")
        self._index = self.client.Index(name)
        return self._index

    async def _ids_with_prefix(self, index: Any, prefix: str, namespace: str) -> list[str]:
        def collect() -> list[str]:
            ids: list[str] = []
            for page in index.list(prefix=prefix, namespace=namespace):
                ids.extend(page)
            return ids

        return await self._call(collect)

    async def _delete_ids(self, index: Any, ids: list[str], namespace: str) -> None:
        for start in range(0, len(ids), 1000):
            await self._call(index.delete, ids=ids[start : start + 1000], namespace=namespace)

    async def _empty(self, index: Any) -> None:
        for namespace in NAMESPACES:
            try:
                await self._call(index.delete, delete_all=True, namespace=namespace)
            except Exception as error:
                if "not found" not in str(error).lower() and "404" not in str(error):
                    raise

    async def upsert(self, documents: list[Document], replace_all: bool = False) -> dict[str, int]:
        passages = [passage for document in documents for passage in to_passages(document)]
        vectors: list[list[float]] = []
        for start in range(0, len(passages), EMBED_BATCH):
            vectors.extend(await self.embeddings.aembed_documents([passage.text for passage in passages[start : start + EMBED_BATCH]]))
        dimension = len(vectors[0]) if vectors else len(await self.embeddings.aembed_query("size"))
        index = await self.index(dimension)

        if replace_all:
            await self._empty(index)
        else:
            # Removes the record's old passages first, since an edit can leave it shorter than before
            for document in documents:
                stale = await self._ids_with_prefix(index, id_prefix(document.type, document.id, document.locale), document.locale)
                await self._delete_ids(index, stale, document.locale)

        by_namespace: dict[str, list[dict[str, Any]]] = {namespace: [] for namespace in NAMESPACES}
        for passage, vector in zip(passages, vectors, strict=True):
            by_namespace[passage.locale].append({"id": passage.id, "values": vector, "metadata": passage.metadata()})
        for namespace, items in by_namespace.items():
            for start in range(0, len(items), UPSERT_BATCH):
                await self._call(index.upsert, vectors=items[start : start + UPSERT_BATCH], namespace=namespace)
        return {"documents": len(documents), "passages": len(passages), "dimension": dimension}

    async def delete(self, items: list[IndexItem]) -> int:
        index = await self.index()
        removed = 0
        for item in items:
            for namespace in NAMESPACES:
                ids = await self._ids_with_prefix(index, f"{item.type}:{item.id}:", namespace)
                await self._delete_ids(index, ids, namespace)
                removed += len(ids)
        return removed

    async def _query(self, index: Any, vector: list[float], namespace: str, top_k: int) -> list[Passage]:
        result = await self._call(index.query, vector=vector, top_k=top_k, namespace=namespace, include_metadata=True)
        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
        passages: list[Passage] = []
        for match in matches or []:
            get = match.get if isinstance(match, dict) else lambda key, default=None, m=match: getattr(m, key, default)
            metadata = get("metadata") or {}
            score = float(get("score") or 0)
            if score < MIN_SCORE:
                continue
            passages.append(
                Passage(
                    id=str(get("id")),
                    text=str(metadata.get("text", "")),
                    type=str(metadata.get("type", "")),
                    record_id=str(metadata.get("record_id", "")),
                    locale=namespace,
                    title=str(metadata.get("title", "")),
                    url=str(metadata.get("url", "")),
                    score=score,
                )
            )
        return passages

    async def search(self, query: str, locale: str, top_k: int = 6) -> list[Passage]:
        index = await self.index()
        vector = await self.embeddings.aembed_query(query)
        found = await self._query(index, vector, locale, top_k)
        # Few matches in the visitor's language: the other language may still hold the answer
        if len(found) < 3:
            other = "bn" if locale == "en" else "en"
            found += await self._query(index, vector, other, top_k - len(found))
        return found[:top_k]


async def store_for(retrieval: Retrieval) -> PineconeStore | None:
    if retrieval.pinecone and retrieval.embedding:
        return PineconeStore(retrieval.pinecone, await build_embeddings(retrieval.embedding))
    return None


async def retrieve(retrieval: Retrieval, query: str, locale: str, top_k: int = 6) -> list[Passage]:
    """Searches Pinecone when it is set up, otherwise the records sent with the request."""
    store = await store_for(retrieval)
    if store is not None:
        return await store.search(query, locale, top_k)
    return keyword_search(retrieval.documents, query, locale, top_k)
