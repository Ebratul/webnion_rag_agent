from app.graphs.site_assistant import guard
from app.schemas import ChatRequest
from tests.conftest import DOCUMENTS, FAKE_CHAT, events, post


def chat(client, messages, **extra):
    payload = {"messages": messages, "chat_model": FAKE_CHAT, "retrieval": {"documents": DOCUMENTS}, **extra}
    return events(post(client, "/v1/chat", payload))


def text(found):
    return "".join(event["text"] for event in found if event["event"] == "token")


def test_answer_streams_with_sources(client):
    found = chat(client, [{"role": "user", "content": "How much is the ecommerce starter plan?"}])
    assert "499" in text(found)
    sources = next(event for event in found if event["event"] == "sources")
    assert sources["items"][0] == {"title": "Starter plan", "url": "/en#pricing"}
    assert any(event["event"] == "usage" for event in found)
    assert found[-1]["event"] == "done"


def test_inquiry_is_only_proposed(client):
    found = chat(client, [{"role": "user", "content": "I want an app. My email is Karim@Example.com"}])
    proposal = next(event for event in found if event["event"] == "inquiry")["proposal"]
    assert proposal["email"] == "karim@example.com"
    assert text(found)


def test_no_inquiry_when_switched_off(client):
    found = chat(client, [{"role": "user", "content": "Contact me at karim@example.com"}], assistant={"allow_inquiry": False})
    assert not any(event["event"] == "inquiry" for event in found)


def test_no_inquiry_without_tool_calling(client):
    payload_model = {**FAKE_CHAT, "tools": False}
    found = events(post(client, "/v1/chat", {"messages": [{"role": "user", "content": "mail karim@example.com"}], "chat_model": payload_model}))
    assert not any(event["event"] == "inquiry" for event in found)


def test_repeated_message_is_stopped(client):
    turns = []
    for _ in range(3):
        turns += [{"role": "user", "content": "price?"}, {"role": "assistant", "content": "..."}]
    found = chat(client, turns[:-1])
    assert "same message" in text(found)
    assert not any(event["event"] == "usage" for event in found)


def test_bangla_question_is_answered_in_bangla():
    request = ChatRequest.model_validate({"locale": "en", "messages": [{"role": "user", "content": "আপনারা কি মোবাইল অ্যাপ বানান?"}], "chat_model": FAKE_CHAT})
    assert guard({"request": request})["language"] == "bn"


def test_long_message_is_refused(client):
    response = post(client, "/v1/chat", {"messages": [{"role": "user", "content": "x" * 1600}], "chat_model": FAKE_CHAT})
    assert response.status_code == 422


def test_unsafe_links_are_dropped_from_documents():
    from app.schemas import Document

    document = Document(type="faq", id="a", locale="en", title="t", text="x", url="javascript:alert(1)")
    assert document.url == ""
    assert Document(type="faq", id="a", locale="en", title="t", text="x", url="//evil.example").url == ""
