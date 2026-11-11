"""Turns provider errors into a short code and a plain message, with keys removed."""

import re

from app.providers.safe_url import UnsafeUrlError

_SECRET_PATTERNS = [
    re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{8,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),
    re.compile(r"\bpcsk_[A-Za-z0-9_]{8,}"),
    re.compile(r"\blsv2_[A-Za-z0-9_]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
]
# These keep the label in front of the secret
_LABELLED_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"),
]

MESSAGES = {
    "auth": "The API key was refused. Check that it is correct and active.",
    "model": "The provider does not know this model name, or the key cannot use it.",
    "rate_limit": "The provider's rate limit or quota was reached. Try again later or check the account's billing.",
    "unavailable": "The provider did not answer in time or is down. Try again shortly.",
    "loading": "The model is still loading on Hugging Face. Try again in a minute.",
    "bad_request": "The provider refused the request.",
    "unsafe_url": "The base address must be a public https:// address.",
    "not_supported": "This provider or model cannot do this job.",
    "content_filter": "The provider refused this request under its content rules.",
    "error": "The AI provider returned an error.",
}


class ProviderError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        message = MESSAGES.get(self.code, MESSAGES["error"])
        if self.detail:
            message = f"{message} ({self.detail})"
        return {"code": self.code, "message": message}


def redact(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[hidden]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[hidden]", text)
    for pattern in _LABELLED_PATTERNS:
        text = pattern.sub(r"\1[hidden]", text)
    return text


def _status(error: BaseException) -> int | None:
    for attribute in ("status_code", "status", "http_status", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify(error: BaseException, *secrets: str) -> ProviderError:
    """Sorts any exception from a provider SDK into one of the codes above."""
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, UnsafeUrlError):
        return ProviderError("unsafe_url", str(error))

    name = type(error).__name__.lower()
    text = str(error).lower()
    status = _status(error)

    if (
        status in (401, 403)
        or "authentication" in name
        or "permissiondenied" in name
        or "invalid api key" in text
        or "api key not valid" in text
        or "incorrect api key" in text
        or "unauthorized" in text
    ):
        code = "auth"
    elif status == 404 or "notfound" in name or "model_not_found" in text or "does not exist" in text or "not found" in text or "unknown model" in text:
        code = "model"
    elif status == 429 or "ratelimit" in name or "rate limit" in text or "quota" in text or "resource_exhausted" in text:
        code = "rate_limit"
    elif status == 503 and "loading" in text:
        code = "loading"
    elif "content_policy" in text or "safety" in text or "moderation" in text:
        code = "content_filter"
    elif status in (500, 502, 503, 504, 529) or "timeout" in name or "timed out" in text or "connect" in name or "overloaded" in text or "connection" in text:
        code = "unavailable"
    elif status in (400, 422) or "badrequest" in name or "invalid" in text:
        code = "bad_request"
    else:
        code = "error"

    # A short, cleaned detail helps the admin when the provider says something specific, such as a wrong parameter
    detail = ""
    if code in ("bad_request", "model", "error"):
        detail = redact(str(error), *secrets).replace("\n", " ")[:240]
    return ProviderError(code, detail)
