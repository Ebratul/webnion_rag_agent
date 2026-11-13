import base64

from app.graphs.blog_writer import parse_article, slugify
from tests.conftest import DOCUMENTS, FAKE_CHAT, events, post

BRIEF = {"topic": "Why a fast website sells more", "length": "short", "keywords": ["page speed"]}


def outline(client, structured=True):
    response = post(
        client, "/v1/blog/outline", {"brief": BRIEF, "writing_model": {**FAKE_CHAT, "structured": structured}, "retrieval": {"documents": DOCUMENTS}}
    )
    assert response.status_code == 200, response.text
    return response.json()["outline"]


def test_outline_with_and_without_structured_output(client):
    assert len(outline(client, structured=True)["sections"]) >= 2
    assert len(outline(client, structured=False)["sections"]) >= 2


def test_write_returns_a_draft_and_picture_plan(client):
    plan = outline(client)
    found = events(post(client, "/v1/blog/write", {"brief": BRIEF, "outline": plan, "writing_model": FAKE_CHAT, "max_pictures": 2}))
    assert any(event["event"] == "progress" for event in found)
    result = next(event for event in found if event["event"] == "result")
    headings = [block for block in result["post"]["blocks"] if block["type"] == "heading"]
    assert len(headings) == len(plan["sections"])
    assert result["post"]["slug"]
    assert 1 <= len(result["pictures"]) <= 2
    assert result["pictures"][0]["slot"] == "cover"
    # Every prompt carries the shared art direction and the quality rules, whatever the model wrote
    assert all("Art direction:" in picture["prompt"] and "No text, letters" in picture["prompt"] for picture in result["pictures"])
    assert all(len(picture["prompt"]) <= 1900 for picture in result["pictures"])


def test_translate_keeps_pictures(client):
    post_draft = {
        "title": "Fast sites",
        "excerpt": "Speed matters",
        "blocks": [
            {"type": "heading", "text": "Why speed"},
            {"type": "image", "src": "https://blob.example/a.webp", "alt": "chart"},
            {"type": "paragraph", "text": "Visitors leave slow pages."},
        ],
        "tags": ["speed"],
    }
    found = events(post(client, "/v1/blog/translate", {"writing_model": FAKE_CHAT, "post": post_draft}))
    result = next(event for event in found if event["event"] == "result")
    assert any(block["type"] == "image" for block in result["post"]["blocks"])


def test_picture_is_webp(client):
    response = post(client, "/v1/images", {"image_model": {"provider": "fake", "model": "fake"}, "prompt": "an abstract shape"})
    assert response.status_code == 200
    data = base64.b64decode(response.json()["data"])
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def test_model_test_reports_capabilities(client):
    chat = post(client, "/v1/models/test", {"kind": "chat", "model": {"provider": "fake", "model": "fake"}}).json()
    assert chat["ok"] and chat["answers"] and chat["tools"] is not None
    embedding = post(client, "/v1/models/test", {"kind": "embedding", "model": {"provider": "fake", "model": "fake"}}).json()
    assert embedding["dimension"] == 384


def test_model_test_refuses_the_wrong_job(client):
    response = post(client, "/v1/models/test", {"kind": "image", "model": {"provider": "anthropic", "model": "claude-opus-5-5", "api_key": "x"}})
    assert response.status_code == 422


def test_model_test_reports_unsafe_base_address(client):
    result = post(
        client,
        "/v1/models/test",
        {"kind": "chat", "model": {"provider": "openai_compatible", "model": "m", "api_key": "k", "base_url": "https://127.0.0.1/v1"}},
    ).json()
    assert result["ok"] is False and result["error"]["code"] == "unsafe_url"


def test_search_endpoint(client):
    response = post(client, "/v1/search", {"query": "ecommerce price", "retrieval": {"documents": DOCUMENTS}})
    assert response.json()["results"][0]["title"] == "Starter plan"


def test_article_parsing():
    blocks = parse_article("## One\n\nFirst **bold** line\ncontinues.\n\n- a\n- b\n\n### Two\n\n<script>x</script>Text")
    assert blocks == [
        {"type": "heading", "text": "One"},
        {"type": "paragraph", "text": "First bold line continues."},
        {"type": "list", "items": ["a", "b"]},
        {"type": "heading", "text": "Two"},
        {"type": "paragraph", "text": "xText"},
    ]
    assert slugify("Why a Fast Website Sells More!") == "why-a-fast-website-sells-more"


def test_providers_build_without_network():
    import asyncio

    from app.providers.chat import build_chat_model
    from app.schemas import ChatModelConfig

    for provider, model in [
        ("openai", "gpt-5-mini"),
        ("anthropic", "claude-haiku-5-5"),
        ("google_genai", "gemini-2.5-flash"),
        ("groq", "llama-3.3-70b-versatile"),
        ("mistralai", "mistral-small-latest"),
        ("huggingface", "Qwen/Qwen3-8B"),
    ]:
        built = asyncio.run(build_chat_model(ChatModelConfig(provider=provider, model=model, api_key="test-key")))
        assert built is not None
        assert "temperature" not in {key for key, value in built.__dict__.items() if value is not None and key == "temperature"} or provider != "anthropic"


def test_pictures_are_planned_from_what_each_section_says():
    from app.graphs.blog_writer import full_prompt, section_excerpts
    from app.graphs.prompts import pictures_prompt

    article = "## Trust\n\nCustomers in Dhaka check reviews before paying with bKash.\n\n- One\n\n## Speed\n\nSlow pages lose buyers on mobile data."
    excerpts = section_excerpts(article, 3)
    assert excerpts[0].startswith("Customers in Dhaka check reviews")
    assert excerpts[1] == "Slow pages lose buyers on mobile data."
    assert excerpts[2] == ""
    prompt = pictures_prompt("Five mistakes", "Practical fixes", "New founders", list(zip(["Trust", "Speed", "Growth"], excerpts, strict=True)), 3, "")
    assert "Customers in Dhaka check reviews" in prompt and "art director" in prompt
    assert full_prompt("A shop owner at a counter.", "Warm film photography.").startswith(
        "A shop owner at a counter. Art direction: Warm film photography. Premium editorial quality"
    )
