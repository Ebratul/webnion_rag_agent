"""LangSmith tracing for one request, only when the admin has added a LangSmith key.

Traces never include API keys, and email addresses and phone numbers in messages are masked before they leave.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from langsmith import Client
from langsmith.run_helpers import tracing_context

from app.providers.errors import redact
from app.schemas import LangSmithConfig

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s()-]{6,}\d")
_SECRET_KEYS = {"api_key", "apikey", "authorization", "token", "password", "secret"}


def scrub(value: Any) -> Any:
    if isinstance(value, str):
        return _PHONE.sub("[phone]", _EMAIL.sub("[email]", redact(value)))
    if isinstance(value, dict):
        return {key: "[hidden]" if str(key).lower() in _SECRET_KEYS else scrub(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(item) for item in value]
    return value


@contextmanager
def traced(config: LangSmithConfig | None, feature: str, *, tags: list[str] | None = None, metadata: dict[str, str] | None = None) -> Iterator[None]:
    if config is None:
        with tracing_context(enabled=False):
            yield
        return
    client = Client(api_key=config.api_key.get_secret_value(), hide_inputs=scrub, hide_outputs=scrub, hide_metadata=scrub)
    try:
        with tracing_context(enabled=True, client=client, project_name=config.project, tags=[feature, *(tags or [])], metadata=metadata or {}):
            yield
    finally:
        # A serverless function can be frozen right after it answers, so traces are sent before that
        with suppress(Exception):
            client.flush()
