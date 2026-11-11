"""Checks that every request comes from the Express API.

The Express API signs each request with HMAC-SHA256 over

    "{timestamp}.{nonce}.{METHOD}.{path}.{sha256(body) in hex}"

using AGENT_SHARED_SECRET, and sends the result in three headers. A request is refused when the signature is missing or
wrong, when the timestamp is outside the allowed window, or when its nonce was already used within that window.
"""

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict

from fastapi import HTTPException, Request

from app.config import get_settings

TIMESTAMP_HEADER = "x-webnion-timestamp"
NONCE_HEADER = "x-webnion-nonce"
SIGNATURE_HEADER = "x-webnion-signature"

_NONCE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_SIGNATURE = re.compile(r"^[0-9a-f]{64}$")
_MAX_NONCES = 50_000


class NonceCache:
    """Remembers recent nonces so a captured request cannot be sent again.

    Each running copy of the service keeps its own list. A replay that lands on another copy is still limited by the
    timestamp window, and every route is safe to repeat apart from cost.
    """

    def __init__(self, max_items: int = _MAX_NONCES) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_items

    def use(self, nonce: str, expires_at: float) -> bool:
        now = time.time()
        with self._lock:
            while self._seen:
                oldest, expiry = next(iter(self._seen.items()))
                if expiry > now and len(self._seen) < self._max:
                    break
                self._seen.pop(oldest)
            if nonce in self._seen:
                return False
            self._seen[nonce] = expires_at
            return True


nonces = NonceCache()


def signature_for(secret: str, timestamp: str, nonce: str, method: str, path: str, body: bytes) -> str:
    message = f"{timestamp}.{nonce}.{method.upper()}.{path}.{hashlib.sha256(body).hexdigest()}"
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def _refuse(status: int = 401, message: str = "Request not signed correctly") -> HTTPException:
    return HTTPException(status_code=status, detail={"code": "unauthorized" if status == 401 else "bad_request", "message": message})


async def verify_signature(request: Request) -> None:
    """FastAPI dependency for every route except /health."""
    settings = get_settings()
    secret = settings.agent_shared_secret.get_secret_value()
    if not secret:
        # Failing closed: a service without a secret answers nothing
        raise HTTPException(status_code=503, detail={"code": "not_configured", "message": "AGENT_SHARED_SECRET is not set"})

    length = request.headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > settings.max_body_bytes):
        raise _refuse(413, "Request is too large")

    timestamp = request.headers.get(TIMESTAMP_HEADER, "")
    nonce = request.headers.get(NONCE_HEADER, "")
    signature = request.headers.get(SIGNATURE_HEADER, "").lower()
    if not timestamp.isdigit() or len(timestamp) > 12 or not _NONCE.match(nonce) or not _SIGNATURE.match(signature):
        raise _refuse()

    skew = settings.allowed_clock_skew_seconds
    sent_at = int(timestamp)
    if abs(time.time() - sent_at) > skew:
        raise _refuse(message="Request expired")

    body = await request.body()
    if len(body) > settings.max_body_bytes:
        raise _refuse(413, "Request is too large")

    expected = signature_for(secret, timestamp, nonce, request.method, request.url.path, body)
    if not hmac.compare_digest(expected, signature):
        raise _refuse()

    # Checked last, so unsigned requests cannot fill the cache
    if not nonces.use(nonce, sent_at + skew):
        raise _refuse(message="Request already used")
