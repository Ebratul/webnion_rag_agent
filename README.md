# Webnion AI Agent

The AI service behind the Webnion website: the chat assistant on the site and in the admin panel, the blog writer in AI
Studio, picture generation and the Pinecone search index. Python, FastAPI, LangChain, LangGraph, LangSmith and Pinecone.
The full design is in [PLAN.md](PLAN.md).

The service is never called by browsers. Only the Express API (`webnion_backend`) calls it, and every request must carry
an HMAC signature made with `AGENT_SHARED_SECRET`. It stores nothing: provider keys, model names and settings arrive
with each signed request, decrypted by the Express API from AI Settings.

## Run locally

```bash
uv sync
cp .env.example .env            # set AGENT_SHARED_SECRET to the same value as in webnion_backend/.env
uv run uvicorn app.main:app --port 8000
```

In `webnion_backend/.env` set `AGENT_URL=http://127.0.0.1:8000`, the same `AGENT_SHARED_SECRET`, and an `AI_SECRETS_KEY`.
With `AGENT_ALLOW_FAKE_PROVIDER=1` here and `AI_ALLOW_FAKE=1` there, AI Settings offers a "Fake" provider that answers
without calling any AI service, so the whole flow can be tried for free. Never set either on Vercel.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

The tests use a fake model, fake embeddings and a fake Pinecone, so they cost nothing.

## Evaluation

`evals/questions.json` holds 30 questions in English and Bangla with the facts a right answer must contain, plus
off-topic and prompt-injection questions it must refuse. Run it with a real model whenever prompts or models change:

```bash
curl -H "Authorization: Bearer $TOKEN" "$API/ai/index/documents" | jq .data > evals/documents.json
EVAL_PROVIDER=openai EVAL_MODEL=gpt-5-mini EVAL_API_KEY=... uv run python -m evals.run
```

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, unsigned |
| `POST /v1/models/test` | The Test button: answers, tool calling, structured output, embedding size, pictures |
| `POST /v1/chat` | Site assistant, streamed as Server-Sent Events |
| `POST /v1/blog/outline` | Blog step 1 |
| `POST /v1/blog/write` | Blog step 2, streamed: article, edit pass, SEO fields and a picture plan |
| `POST /v1/blog/translate` | Bangla version of a draft, streamed |
| `POST /v1/images` | One picture, returned as WebP |
| `POST /v1/index/upsert`, `/v1/index/delete` | Pinecone updates |
| `POST /v1/search` | Search test from AI Settings |

## Security

- Signed requests: HMAC-SHA256 over the timestamp, a nonce, the method, the path and the body's hash. Requests older than
  five minutes, reused nonces, changed bodies and bodies over 4 MB are refused before any validation runs.
- Without `AGENT_SHARED_SECRET` the service refuses every request. Validation errors never echo submitted values.
- Base addresses for OpenAI-compatible providers and Hugging Face endpoints must be public `https://` addresses; host
  names are resolved and private, loopback, link-local and metadata addresses refused.
- Provider errors are reduced to a code and a plain message, with keys removed.
- The public assistant's only action is to propose an inquiry; the visitor checks it and sends it through the normal
  inquiry form endpoint. Site content and visitor text are passed as marked data, never as instructions.
- LangSmith tracing runs only with a key from AI Settings, and masks emails, phone numbers and keys.

## Deploy

A Vercel project with this folder as its root. `vercel.json` sets the Singapore region; Vercel's FastAPI preset sends every path to the app, and functions get five minutes by default.
Set `AGENT_SHARED_SECRET` in the project's environment variables; provider, Pinecone and LangSmith keys are entered in the
admin panel, not here.

## Working rules

Python 3.12 or newer with `uv`. Code passes `ruff` and `pytest` before every push. Commits use the `Ebratul` identity with a
summary line and a body. No `Co-Authored-By` or "Generated with" lines.
