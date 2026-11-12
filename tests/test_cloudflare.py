import base64
import io
import json

import httpx
import pytest
from PIL import Image

from app.providers import images
from app.providers.errors import ProviderError
from app.schemas import ImageModelConfig
from tests.conftest import post

ACCOUNT = "0123456789abcdef0123456789abcdef"


def png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 36), (120, 60, 200)).save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def cloudflare(monkeypatch):
    seen: list[httpx.Request] = []

    def use(handler):
        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(images, "_cloudflare_transport", httpx.MockTransport(record))
        return seen

    return use


def config(model: str = "@cf/black-forest-labs/flux-1-schnell") -> ImageModelConfig:
    return ImageModelConfig(provider="cloudflare", model=model, api_key="cf-token-123", account_id=ACCOUNT)


async def test_flux_answer_in_json(cloudflare):
    seen = cloudflare(lambda request: httpx.Response(200, json={"success": True, "result": {"image": base64.b64encode(png()).decode()}}))
    picture = await images.generate_picture(config(), "an abstract shape", "landscape")
    assert picture.data[:4] == b"RIFF"
    request = seen[0]
    assert str(request.url) == f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    assert request.headers["authorization"] == "Bearer cf-token-123"
    assert json.loads(request.content) == {"prompt": "an abstract shape", "steps": 8}


async def test_stable_diffusion_answer_as_a_picture(cloudflare):
    seen = cloudflare(lambda request: httpx.Response(200, content=png(), headers={"content-type": "image/png"}))
    picture = await images.generate_picture(config("@cf/stabilityai/stable-diffusion-xl-base-1.0"), "a calm office", "square")
    assert picture.data[:4] == b"RIFF"
    assert json.loads(seen[0].content) == {"prompt": "a calm office", "width": 1024, "height": 1024}


async def test_wrong_token_is_reported_as_auth(cloudflare):
    cloudflare(lambda request: httpx.Response(401, json={"success": False, "errors": [{"code": 10000, "message": "Authentication error"}]}))
    with pytest.raises(ProviderError) as error:
        await images.generate_picture(config(), "x y z", "landscape")
    assert error.value.code == "auth"


async def test_unsafe_model_names_never_reach_the_network(cloudflare):
    seen = cloudflare(lambda request: httpx.Response(200))
    for model in ["@cf/../../user/tokens", "flux-1-schnell", "@cf/a/b/c", "@cf/a/..b/../c"]:
        with pytest.raises(ProviderError):
            await images.generate_picture(config(model), "x y z", "landscape")
    assert seen == []


def test_account_id_is_required_and_checked(client):
    model = {"provider": "cloudflare", "model": "@cf/black-forest-labs/flux-1-schnell", "api_key": "token-123"}
    assert post(client, "/v1/images", {"image_model": model, "prompt": "an abstract shape"}).status_code == 422
    bad = {**model, "account_id": "../../evil"}
    assert post(client, "/v1/images", {"image_model": bad, "prompt": "an abstract shape"}).status_code == 422
    chat = post(client, "/v1/models/test", {"kind": "chat", "model": {**model, "account_id": ACCOUNT}})
    assert chat.status_code == 422
