from typing import Any

from app.providers.fake import HashingEmbeddings
from app.rag.documents import keyword_search, to_passages
from app.rag.store import PineconeStore
from app.schemas import Document, IndexItem, PineconeConfig
from tests.conftest import DOCUMENTS


def test_passages_carry_ids_and_titles():
    document = Document(type="case-study", id="abc", locale="en", title="Shop rebuild", url="/en/case-studies/shop", text="word " * 2000)
    passages = to_passages(document)
    assert len(passages) > 1
    assert passages[0].id == "case-study:abc:en:0"
    assert passages[1].text.startswith("Shop rebuild")


def test_keyword_search_finds_pricing():
    found = keyword_search([Document(**item) for item in DOCUMENTS], "ecommerce price starter", "en")
    assert found[0].title == "Starter plan"


def test_bangla_words_stay_whole():
    from app.rag.documents import _tokens

    assert _tokens("আপনারা কি মোবাইল অ্যাপ বানান?") == ["আপনারা", "কি", "মোবাইল", "অ্যাপ", "বানান"]


def test_keyword_search_in_bangla():
    found = keyword_search([Document(**item) for item in DOCUMENTS], "মোবাইল অ্যাপ", "bn")
    assert found[0].locale == "bn"


class FakeIndex:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, dict[str, Any]]] = {}

    def upsert(self, vectors, namespace):
        for vector in vectors:
            self.data.setdefault(namespace, {})[vector["id"]] = vector

    def list(self, prefix, namespace):
        yield [key for key in self.data.get(namespace, {}) if key.startswith(prefix)]

    def delete(self, ids=None, namespace="", delete_all=False):
        if delete_all:
            if namespace not in self.data:
                raise RuntimeError("Namespace not found")
            self.data.pop(namespace)
            return
        for key in ids or []:
            self.data.get(namespace, {}).pop(key, None)

    def query(self, vector, top_k, namespace, include_metadata):
        import math

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

        scored = sorted(
            ({"id": key, "score": cosine(vector, item["values"]), "metadata": item["metadata"]} for key, item in self.data.get(namespace, {}).items()),
            key=lambda m: -m["score"],
        )
        return {"matches": scored[:top_k]}


class FakePinecone:
    def __init__(self) -> None:
        self.created: dict[str, int] = {}
        self.indexes: dict[str, FakeIndex] = {}

    def has_index(self, name):
        return name in self.created

    def describe_index(self, name):
        return type("Described", (), {"dimension": self.created[name], "status": {"ready": True}})()

    def create_index(self, name, dimension, metric, spec):
        assert metric == "cosine"
        self.created[name] = dimension

    def Index(self, name):
        return self.indexes.setdefault(name, FakeIndex())


CONFIG = PineconeConfig(api_key="pc-test", index="webnion")


async def test_store_creates_updates_and_deletes():
    client = FakePinecone()
    embeddings = HashingEmbeddings(16)
    store = PineconeStore(CONFIG, embeddings, client=client)
    long = Document(type="service", id="web", locale="en", title="Web", url="/en/services/web", text="word " * 2000)
    result = await store.upsert([long], replace_all=True)
    assert client.created == {"webnion": 16}
    assert result["passages"] > 1

    short = long.model_copy(update={"text": "Now a short page"})
    await PineconeStore(CONFIG, embeddings, client=client).upsert([short])
    assert list(client.indexes["webnion"].data["en"]) == ["service:web:en:0"]

    found = await PineconeStore(CONFIG, embeddings, client=client).search("Web\n\nNow a short page", "en")
    assert found and found[0].url == "/en/services/web"

    removed = await PineconeStore(CONFIG, embeddings, client=client).delete([IndexItem(type="service", id="web")])
    assert removed == 1


async def test_store_refuses_a_dimension_mismatch():
    import pytest

    from app.providers.errors import ProviderError

    client = FakePinecone()
    client.created["webnion"] = 1536
    store = PineconeStore(CONFIG, HashingEmbeddings(16), client=client)
    with pytest.raises(ProviderError):
        await store.upsert([Document(type="faq", id="a", locale="en", title="t", text="hello")])
