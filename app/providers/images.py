"""Picture generation. Every picture comes back as a WebP, small enough for Vercel's response limit."""

import asyncio
import base64
import io
import json
import re
from dataclasses import dataclass

import httpx
from PIL import Image, ImageDraw

from app.providers.errors import ProviderError
from app.providers.fake import require_fake_allowed
from app.providers.safe_url import ensure_public_url
from app.schemas import ModelConfig

MAX_SIDE = 1600
MAX_DOWNLOAD = 25 * 1024 * 1024
# Pictures from providers are a few megapixels. This refuses anything absurd before it is decoded.
Image.MAX_IMAGE_PIXELS = 40_000_000


@dataclass
class Picture:
    data: bytes
    width: int
    height: int
    mime: str = "image/webp"

    def as_dict(self) -> dict[str, object]:
        return {"mime": self.mime, "width": self.width, "height": self.height, "data": base64.b64encode(self.data).decode()}


def to_webp(raw: bytes | Image.Image) -> Picture:
    image = raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw))
    image.load()
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    image.save(out, format="WEBP", quality=82, method=4)
    return Picture(out.getvalue(), image.width, image.height)


def _openai_size(model: str, aspect: str) -> str:
    if model.startswith("dall-e-2"):
        return "1024x1024"
    if model.startswith("dall-e-3"):
        return "1792x1024" if aspect == "landscape" else "1024x1024"
    return "1536x1024" if aspect == "landscape" else "1024x1024"


async def _download(url: str) -> bytes:
    await ensure_public_url(url)
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client, client.stream("GET", url) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_DOWNLOAD:
                raise ProviderError("bad_request", "picture too large")
            chunks.append(chunk)
        return b"".join(chunks)


async def _openai(config: ModelConfig, prompt: str, aspect: str) -> bytes:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.api_key.get_secret_value(), timeout=180, max_retries=1)
    extra = {"response_format": "b64_json"} if config.model.startswith("dall-e") else {}
    result = await client.images.generate(model=config.model, prompt=prompt, size=_openai_size(config.model, aspect), n=1, **extra)
    item = result.data[0] if result.data else None
    if item is None:
        raise ProviderError("error", "no picture returned")
    if item.b64_json:
        return base64.b64decode(item.b64_json)
    if item.url:
        return await _download(item.url)
    raise ProviderError("error", "no picture returned")


async def _google(config: ModelConfig, prompt: str, aspect: str) -> bytes:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.api_key.get_secret_value())
    ratio = "16:9" if aspect == "landscape" else "1:1"
    if config.model.startswith("imagen"):
        result = await client.aio.models.generate_images(
            model=config.model, prompt=prompt, config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=ratio)
        )
        images = result.generated_images or []
        if images and images[0].image and images[0].image.image_bytes:
            return images[0].image.image_bytes
        raise ProviderError("content_filter", "no picture returned")

    # Gemini picture models answer through generate_content with picture output
    result = await client.aio.models.generate_content(
        model=config.model,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio=ratio)),
    )
    for candidate in result.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    raise ProviderError("content_filter", "no picture returned")


async def _huggingface(config: ModelConfig, prompt: str, aspect: str) -> Image.Image:
    from huggingface_hub import AsyncInferenceClient

    width, height = (1344, 768) if aspect == "landscape" else (1024, 1024)
    key = config.api_key.get_secret_value()
    if config.base_url:
        client = AsyncInferenceClient(model=await ensure_public_url(config.base_url), api_key=key, timeout=180)
        return await client.text_to_image(prompt, width=width, height=height)
    client = AsyncInferenceClient(provider=config.hf_provider or "auto", api_key=key, timeout=180)  # type: ignore[arg-type]
    return await client.text_to_image(prompt, model=config.model, width=width, height=height)


# Cloudflare Workers AI. The address is fixed: only the account ID and a checked model name go into it.
CLOUDFLARE_API = "https://api.cloudflare.com/client/v4/accounts"
_CLOUDFLARE_MODEL = re.compile(r"^@(cf|hf)/[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*$")
# Tests swap in a fake transport
_cloudflare_transport: httpx.AsyncBaseTransport | None = None


def cloudflare_model_ok(model: str) -> bool:
    return bool(_CLOUDFLARE_MODEL.match(model)) and ".." not in model


async def _cloudflare(config: ModelConfig, prompt: str, aspect: str) -> bytes:
    if not cloudflare_model_ok(config.model) or not config.account_id:
        raise ProviderError("model", "Cloudflare model names look like @cf/black-forest-labs/flux-1-schnell")
    # FLUX schnell takes only a prompt and up to 8 steps; the Stable Diffusion family takes a size
    if "flux-1-schnell" in config.model:
        body: dict[str, object] = {"prompt": prompt[:2048], "steps": 8}
    else:
        width, height = (1344, 768) if aspect == "landscape" else (1024, 1024)
        body = {"prompt": prompt[:2048], "width": width, "height": height}

    url = f"{CLOUDFLARE_API}/{config.account_id}/ai/run/{config.model}"
    headers = {"Authorization": f"Bearer {config.api_key.get_secret_value()}"}
    async with (
        httpx.AsyncClient(timeout=180, follow_redirects=False, transport=_cloudflare_transport) as client,
        client.stream("POST", url, json=body, headers=headers) as response,
    ):
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_DOWNLOAD:
                raise ProviderError("bad_request", "picture too large")
            chunks.append(chunk)
        data = b"".join(chunks)
        kind = response.headers.get("content-type", "")

    if response.status_code >= 400:
        message = ""
        try:
            errors = json.loads(data).get("errors") or []
            message = "; ".join(str(error.get("message", "")) for error in errors if isinstance(error, dict))
        except (ValueError, AttributeError):
            pass
        raise ProviderError(classify_status(response.status_code), message[:200])
    # Some models send the picture itself, others JSON with it in base64
    if kind.startswith("image/"):
        return data
    try:
        parsed = json.loads(data)
    except ValueError as error:
        raise ProviderError("error", "Cloudflare returned something that is not a picture") from error
    result = parsed.get("result", parsed) if isinstance(parsed, dict) else {}
    image = result.get("image") if isinstance(result, dict) else None
    if not isinstance(image, str):
        raise ProviderError("content_filter", "no picture returned")
    return base64.b64decode(image)


def classify_status(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "model"
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "unavailable"
    return "bad_request"


def _fake(prompt: str, aspect: str) -> Image.Image:
    width, height = (1200, 675) if aspect == "landscape" else (800, 800)
    image = Image.new("RGB", (width, height), (76, 47, 199))
    draw = ImageDraw.Draw(image)
    for row in range(height):
        shade = int(120 * row / height)
        draw.line([(0, row), (width, row)], fill=(76 + shade // 2, 47 + shade // 3, 199 - shade // 3))
    draw.text((40, 40), prompt[:80], fill=(255, 255, 255))
    return image


async def generate_picture(config: ModelConfig, prompt: str, aspect: str = "landscape") -> Picture:
    if config.provider == "fake":
        require_fake_allowed()
        raw: bytes | Image.Image = _fake(prompt, aspect)
    elif config.provider == "openai":
        raw = await _openai(config, prompt, aspect)
    elif config.provider == "google_genai":
        raw = await _google(config, prompt, aspect)
    elif config.provider == "huggingface":
        raw = await _huggingface(config, prompt, aspect)
    elif config.provider == "cloudflare":
        raw = await _cloudflare(config, prompt, aspect)
    else:
        raise ProviderError("not_supported", f"{config.provider} cannot make pictures here")
    try:
        return await asyncio.to_thread(to_webp, raw)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ProviderError("error", "the provider returned something that is not a picture") from error
