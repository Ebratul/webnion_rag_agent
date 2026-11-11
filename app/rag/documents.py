"""Turns site records into passages, and searches passages by keyword when Pinecone is not set up."""

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import Document

# About 800 tokens of English with 100 overlapping. Bangla takes more tokens per character, which stays well within limits.
CHUNK_CHARS = 3000
OVERLAP_CHARS = 400

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=OVERLAP_CHARS, separators=["\n\n", "\n", "। ", ". ", " ", ""])
_WORD = re.compile(r"\w+", re.UNICODE)
_BENGALI = re.compile(r"[ঀ-৿]")


@dataclass
class Passage:
    id: str
    text: str
    type: str
    record_id: str
    locale: str
    title: str
    url: str
    updated_at: str = ""
    content_hash: str = ""
    score: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, str]:
        return {
            "type": self.type,
            "record_id": self.record_id,
            "locale": self.locale,
            "title": self.title,
            "url": self.url,
            "updated_at": self.updated_at,
            "hash": self.content_hash,
            "text": self.text,
        }


def id_prefix(type_: str, record_id: str, locale: str) -> str:
    return f"{type_}:{record_id}:{locale}:"


def to_passages(document: Document) -> list[Passage]:
    text = document.text.strip()
    if not text:
        return []
    digest = hashlib.sha256(f"{document.title}\n{text}".encode()).hexdigest()[:16]
    chunks = _splitter.split_text(text)
    prefix = id_prefix(document.type, document.id, document.locale)
    return [
        Passage(
            id=f"{prefix}{number}",
            # The title travels with every passage, so a passage from the middle of a page still says what it is about
            text=f"{document.title}\n\n{chunk}" if document.title else chunk,
            type=document.type,
            record_id=document.id,
            locale=document.locale,
            title=document.title,
            url=document.url,
            updated_at=document.updated_at,
            content_hash=digest,
        )
        for number, chunk in enumerate(chunks)
    ]


def detect_language(text: str, fallback: str = "en") -> str:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return fallback
    bengali = sum(1 for character in letters if _BENGALI.match(character))
    return "bn" if bengali / len(letters) > 0.3 else "en"


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD.findall(text) if len(token) > 1]


def keyword_search(documents: list[Document], query: str, locale: str, top_k: int = 6) -> list[Passage]:
    """BM25 over the records sent with the request. Used when Pinecone is not set up."""
    passages = [passage for document in documents for passage in to_passages(document)]
    if not passages:
        return []
    query_terms = set(_tokens(query))
    if not query_terms:
        return []

    bags = [Counter(_tokens(passage.text)) for passage in passages]
    average = sum(sum(bag.values()) for bag in bags) / len(bags) or 1
    frequency = Counter(term for bag in bags for term in bag.keys() & query_terms)
    total = len(passages)

    def score(bag: Counter[str]) -> float:
        length = sum(bag.values()) or 1
        value = 0.0
        for term in query_terms:
            count = bag.get(term, 0)
            if count:
                idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
                value += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / average))
        return value

    for passage, bag in zip(passages, bags, strict=True):
        passage.score = score(bag) * (1.15 if passage.locale == locale else 1.0)
    found = sorted((passage for passage in passages if passage.score > 0), key=lambda passage: passage.score, reverse=True)
    return found[:top_k]
