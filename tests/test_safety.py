import socket

import pytest

from app.providers.errors import classify, redact
from app.providers.safe_url import UnsafeUrlError, check_shape, ensure_public_url
from app.tracing import scrub


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",
        "https://localhost/v1",
        "https://127.0.0.1/v1",
        "https://10.0.0.5/v1",
        "https://169.254.169.254/latest",
        "https://[::1]/v1",
        "https://[::ffff:127.0.0.1]/v1",
        "https://user:pass@api.example.com/v1",
        "https://service.internal/v1",
        "https://api.example.com/v1?x=1",
        "ftp://api.example.com",
    ],
)
def test_unsafe_addresses_are_refused(url):
    with pytest.raises(UnsafeUrlError):
        check_shape(url)


def test_public_address_shape_is_accepted():
    assert check_shape("https://api.deepseek.com/v1") == ("api.deepseek.com", 443)


async def test_name_that_resolves_to_a_private_address_is_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 443))])
    with pytest.raises(UnsafeUrlError):
        await ensure_public_url("https://rebind.example.com/v1")


async def test_name_that_resolves_publicly_is_accepted(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.1.1", 443))])
    assert await ensure_public_url("https://api.example.com/v1/") == "https://api.example.com/v1"


def test_keys_are_removed_from_messages():
    text = "Incorrect API key provided: sk-proj-abcdefghijklmnop. Bearer abcdefghijkl hf_abcdefghijklmn api_key=supersecretvalue"
    cleaned = redact(text, "mysecretkey123")
    for secret in ("sk-proj-abcdefghijklmnop", "abcdefghijkl", "hf_abcdefghijklmn", "supersecretvalue"):
        assert secret not in cleaned


def test_errors_are_sorted():
    class AuthenticationError(Exception):
        status_code = 401

    class RateLimitError(Exception):
        status_code = 429

    class NotFoundError(Exception):
        status_code = 404

    assert classify(AuthenticationError("bad key sk-abcdefghijklmnopq")).code == "auth"
    assert classify(RateLimitError("slow down")).code == "rate_limit"
    assert classify(NotFoundError("model gpt-x does not exist")).code == "model"
    assert classify(TimeoutError("timed out")).code == "unavailable"
    assert "sk-abc" not in classify(ValueError("invalid sk-abcdefghijklmnopq")).as_dict()["message"]


def test_traces_mask_personal_details_and_keys():
    cleaned = scrub({"messages": ["mail me at rahim@example.com or +880 1711-000000"], "api_key": "sk-abc"})
    assert cleaned["api_key"] == "[hidden]"
    assert "rahim@example.com" not in cleaned["messages"][0]
    assert "1711" not in cleaned["messages"][0]
