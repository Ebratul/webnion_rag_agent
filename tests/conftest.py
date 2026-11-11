import json
import secrets
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

SECRET = "test-secret-" + "x" * 40


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch):
    from app.config import get_settings

    monkeypatch.setenv("AGENT_SHARED_SECRET", SECRET)
    monkeypatch.setenv("AGENT_ALLOW_FAKE_PROVIDER", "1")
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def signed_headers(path: str, body: bytes, *, timestamp: int | None = None, nonce: str | None = None, secret: str = SECRET) -> dict[str, str]:
    from app.security import signature_for

    stamp = str(timestamp if timestamp is not None else int(time.time()))
    once = nonce or secrets.token_urlsafe(18)
    return {
        "content-type": "application/json",
        "x-webnion-timestamp": stamp,
        "x-webnion-nonce": once,
        "x-webnion-signature": signature_for(secret, stamp, once, "POST", path, body),
    }


def post(client: TestClient, path: str, payload: dict[str, Any], **options: Any):
    body = json.dumps(payload).encode()
    return client.post(path, content=body, headers=signed_headers(path, body, **options))


def events(response) -> list[dict[str, Any]]:
    found = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines:
            found.append({"event": lines.get("event"), **json.loads(lines["data"])})
    return found


FAKE_CHAT = {"provider": "fake", "model": "fake-model", "tools": True, "structured": True}

DOCUMENTS = [
    {
        "type": "pricing",
        "id": "starter",
        "locale": "en",
        "title": "Starter plan",
        "url": "/en#pricing",
        "text": "The Starter plan costs 499 USD and includes an ecommerce website with five pages.",
    },
    {
        "type": "service",
        "id": "apps",
        "locale": "en",
        "title": "Mobile apps",
        "url": "/en/services/mobile-apps",
        "text": "We build iOS and Android mobile apps with Flutter and React Native.",
    },
    {
        "type": "service",
        "id": "apps",
        "locale": "bn",
        "title": "মোবাইল অ্যাপ",
        "url": "/bn/services/mobile-apps",
        "text": "আমরা ফ্লাটার দিয়ে iOS ও অ্যান্ড্রয়েড মোবাইল অ্যাপ বানাই।",
    },
]
