# Webnion AI Agent: Plan

The AI service for the Webnion website. It answers visitors' questions about Webnion, takes inquiries from the chat, and helps
the team write blog posts with pictures. It is written in Python with LangChain and LangGraph, traced with LangSmith, and
searches the site's content through Pinecone.

Status: plan agreed on 2026-11-12, with Hugging Face and OpenAI-compatible providers added the same day. Phases 0 to 6 were built on 2026-11-12; real providers and Pinecone are checked once the owner enters keys in AI Settings. Translation runs as its own request (`/v1/blog/translate`) to stay inside the five-minute limit.

## 1. Decisions

Taken with the owner on 2026-11-12:

| Question | Decision |
| --- | --- |
| What the chatbot on the public site does | Answers questions about Webnion from the site's own content (RAG) and can send an inquiry from the chat. It does **not** write blogs or make pictures for visitors, because that costs money per use and invites abuse. |
| Where generated blogs go | A new Blog section on the website (`/en/blog`, `/bn/blog`). The AI writes a **draft**; an admin reviews and edits it, then publishes. |
| Which AI provider | Any provider. In the admin panel the owner picks a provider, types a model name and pastes an API key. LangChain connects to it. Hugging Face is one of the providers, and an OpenAI-compatible option with a base address covers providers not in the list. |
| Where the Python service runs | Vercel, in the same Singapore region as the API. |
| Vector database | Pinecone. |

## 2. What the owner will be able to do from the admin panel

- Turn the whole chatbot on or off.
- Choose where it appears: **on the public site**, **in the admin panel**, or both.
- Turn single features on or off: site Q&A, inquiries from the chat, blog writing, picture generation.
- For each job, choose the provider, type the model name and paste the API key (and, for an OpenAI-compatible provider, its base address):
  - chat model for visitors (fast and cheap is best),
  - writing model for blogs (stronger),
  - picture model,
  - embedding model for Pinecone search.
- Paste the Pinecone key and index name, and optionally a LangSmith key and project name.
- Press **Test** next to each model to check the key and the model name before saving. Test also reports what the model can do
  (section 4.2).
- Set limits: messages per visitor per day, blogs per day, pictures per blog.
- Edit the assistant's greeting, the suggested questions and its tone, in English and Bangla.
- Open **AI Studio**, give a topic, approve or change the outline, and get a full draft with a cover picture and inline pictures,
  saved under Blog.
- Rebuild the search index with one button. It also updates by itself when content changes.
- See usage: messages, blogs, pictures and estimated cost per day.

Every change is written to the existing Activity Log.

## 3. How the pieces fit

```
Visitor's browser ─┐                                         ┌─> LLM providers (OpenAI, Anthropic, Google, Hugging Face, ...)
                   │                                         ├─> Picture providers (OpenAI, Google, Hugging Face)
Admin's browser ───┼─> Next.js site ─> Express API ─> AI service (this repo, FastAPI + LangGraph)
                   │   (webnion_frontend) (webnion_backend)   ├─> Pinecone (search index)
                   │                       │                 └─> LangSmith (traces)
                   │                       └─> MySQL (Aiven): AI settings, blogs, usage, chat leads
                   └─> Vercel Blob (generated pictures, saved by the Express API)
```

Why every call goes through the Express API:

- **One place for keys.** API keys are stored encrypted in MySQL by the Express API and decrypted only there. The AI service
  receives the settings it needs with each request and stores nothing, so it needs no database access of its own.
- **One place for rules.** Login, roles, rate limits, the on/off switches, usage limits and the Activity Log already live there.
- **The AI service is not public.** It accepts only requests signed by the Express API (section 9).

## 4. Choosing models without code changes

The admin panel stores, for each job, a **provider**, a **model name** and an **API key**. The AI service builds the client at
request time:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(settings.model, model_provider=settings.provider, api_key=settings.api_key)
```

| Job | Providers in the first version | Notes |
| --- | --- | --- |
| Chat and writing | `openai`, `anthropic`, `google_genai`, `groq`, `mistralai`, `huggingface`, OpenAI-compatible | Any model name that provider accepts. The **Test** button sends one short message to check it. |
| Pictures | OpenAI, Google, Hugging Face, Cloudflare Workers AI | Claude does not make pictures, so a picture provider is always separate. |
| Embeddings | OpenAI, Google, Cohere, Hugging Face, OpenAI-compatible | Anthropic has no embedding model. The model's vector size must match the Pinecone index (section 6). |

One key can cover every job: a single OpenAI, Google or Hugging Face key runs chat, writing, pictures and embeddings.

### 4.1 Hugging Face

Through the `langchain-huggingface` package and the `huggingface_hub` client, with a Hugging Face access token.

- **Chat and writing**: `ChatHuggingFace` over `HuggingFaceEndpoint`, with the model id the admin types, such as
  `org/model-name`.
- **Embeddings**: `HuggingFaceEndpointEmbeddings`. Multilingual models such as `BAAI/bge-m3` handle Bangla well.
- **Pictures**: `InferenceClient.text_to_image` with a text-to-image model id.
- **Where models run**: on Hugging Face's hosted inference (Inference Providers), or on a dedicated Inference Endpoint whose
  address the admin pastes. Models are never run inside this service: Vercel has no GPU and limits a function's size.
- **Costs**: hosted inference includes a small free allowance and then bills the Hugging Face account. Dedicated endpoints bill
  by the hour while running.
- **Not every model is deployed for inference.** Test tells the admin when a model id exists but cannot be called.

### 4.1.1 Cloudflare Workers AI (pictures)

Added on 2026-11-12 at the owner's request. Pictures come from Cloudflare's REST API with the admin's **account ID** and an
API token with the Workers AI permission, for example `@cf/black-forest-labs/flux-1-schnell` (answers with the picture in
base64 JSON) or `@cf/stabilityai/stable-diffusion-xl-base-1.0` (answers with the picture itself; the size is sent). The
address is fixed to `api.cloudflare.com`; only a 32-character account ID and a model name of the form `@cf/org/model` go
into it, checked by both services. Cloudflare's chat and embedding models can use the OpenAI-compatible provider with the
base address `https://api.cloudflare.com/client/v4/accounts/<account ID>/ai/v1`.

### 4.2 What Test checks

Open models differ in what they support, so Test reports more than "works":

- **Answers**: the key and model reply to a short message.
- **Tool calling**: the model can call the inquiry tool. Without it, the site assistant still answers questions, but offers the
  inquiry form link instead of taking the inquiry in the chat.
- **Structured output**: needed by the blog writer for outlines and SEO fields. Without it, the writer asks for JSON in the
  prompt and validates it, retrying once.
- **Embedding size**, for embedding models, so the index can be created or a rebuild suggested.
- **Picture generation**, for picture models, with one small test picture.

The result is stored with the settings, so the service picks the right path without testing again on every request.

### 4.3 OpenAI-compatible providers

Many providers, such as DeepSeek, OpenRouter, Together, Fireworks and Hugging Face's own router, accept requests in OpenAI's
format at their own address. Choosing **OpenAI-compatible** shows a **base address** field next to the model name and key, and
the service connects with LangChain's OpenAI client pointed at that address. New providers of this kind then need no code
changes.

To stop the base address being used to reach internal systems (server-side request forgery), the Express API accepts only
`https://` addresses that resolve to public IP addresses, refusing `localhost`, private and link-local ranges, and the AI
service checks again before connecting.

Provider details the code must handle:

- **Anthropic Claude.** Current models (for example `claude-opus-5-5`, or `claude-sonnet-5-5` and `claude-haiku-5-5` for
  cheaper, faster chat) reject `temperature`, `top_p` and `top_k`, so these are never sent to Claude. Thinking depth is set
  with the effort setting, not a token budget. A reply can end with the stop reason `refusal`; the service then answers
  politely instead of failing.
- **All providers.** Errors are sorted into wrong key, unknown model, rate limit and provider down. The admin sees a plain
  message, and the visitor sees "The assistant is busy, please try WhatsApp or the contact form".
- **Model names change.** Nothing is hard-coded. The admin types the name, and the Test button confirms it works.
- **Hugging Face models load on demand.** A model that has not been called for a while can answer slowly or report that it is
  loading. The service waits and retries once with a clear progress message, within Vercel's time limit.

## 5. The two assistants (LangGraph)

### 5.1 Site assistant (public site and admin panel)

```
input ─> guard ─> retrieve ─> answer ─┬─> end
                                      └─> collect_lead ─> confirm ─> create_inquiry ─> end
```

- **guard**: rejects empty, very long or repeated messages, and detects English or Bangla. It answers in the visitor's language.
- **retrieve**: searches Pinecone for the six closest passages in that language, then all languages if few are found.
- **answer**: answers only from the retrieved passages and the site's contact details, with links to the pages used. If the
  answer is not there, it says so and offers WhatsApp or the inquiry form. Retrieved text is treated as data, never as
  instructions, which guards against prompt injection hidden in content.
- **collect_lead**: when the visitor wants to start a project, it asks for name, email or phone, and what they need. It shows
  a summary and sends the inquiry **only after the visitor confirms**. The inquiry goes through the existing inquiry endpoint,
  so it appears under Inquiries, is logged and triggers the inquiry email.
- **Memory**: the browser keeps the conversation for the visit and sends the last ten turns with each message. The service
  keeps nothing between requests, which suits Vercel. No LangGraph checkpointer is needed.
- **Streaming**: answers stream word by word over Server-Sent Events.

### 5.2 Blog writer (admin panel, AI Studio)

```
Step 1  topic, audience, tone, length, keywords, languages ─> research_outline ─> outline returned to the admin
Step 2  approved outline ─> write_sections ─> edit ─> seo ─> picture_prompts ─> make_pictures ─> translate? ─> save_draft
```

- Two separate requests instead of one long paused graph. The admin can edit the outline between them, and nothing has to
  stay in memory on a serverless function. This is human-in-the-loop without a checkpointer.
- **write_sections** writes in the site's article format: headings, paragraphs, lists and pictures. That format already
  exists for case studies, so the blog editor in the admin panel can reuse it.
- **edit** checks flow, repetition and claims about Webnion against the retrieved site content. It must not invent prices,
  clients or results.
- **seo** writes the title, URL slug, meta description and tags.
- **make_pictures** makes one cover and one picture per main section, up to the admin's limit. The Express API stores them in
  Vercel Blob, and each gets alt text.
- **translate** runs when the admin asks for both languages and writes the Bangla version from the English draft.
- Progress streams to AI Studio: "Writing section 2 of 5", "Making cover picture".
- The draft is saved as a Blog post with status **Draft**. Nothing is published automatically.
- Vercel allows five minutes per request. A long post with several pictures is split into calls: text first, then one call
  per picture.

## 6. Pinecone search index

- One serverless index, cosine metric. Its dimension comes from the chosen embedding model, set when the index is created.
- **Changing the embedding model needs a rebuild.** The admin panel warns about this and offers Rebuild.
- Records: services, projects, case studies, products, FAQs, pricing plans, team, Website Text sections, contact details
  and published blog posts.
- Each record is split into passages of about 800 tokens with 100 overlapping, using LangChain's text splitter.
- Passage id: `{type}:{id}:{locale}:{n}`. Metadata: type, id, slug, URL, locale, title, updated time and a content hash.
  An unchanged record is skipped.
- Namespaces: `en` and `bn`.
- **Keeping it current**: when an admin saves or deletes a record, the Express API asks the AI service to update or remove
  that record. A nightly Vercel Cron job runs a full check in case an update was missed. Rebuild in the admin panel does
  everything at once.

## 7. Changes in the other two repos

### webnion_backend (Express, MySQL)

- `AiSettings`: one row with the on/off switches, placements, limits, prompts, and the provider, model and **encrypted** key
  for each job. Keys use AES-256-GCM with `AI_SECRETS_KEY` from the environment. They are never sent back to the browser,
  only their last four characters.
- `BlogPost`: slug, title, excerpt, content (the article block format), cover, tags, SEO title and description, status
  (Draft or Published), publish date, author, whether AI wrote it, and which models and settings it used.
- `AiUsage`: per day and feature, the number of requests, tokens, pictures and an estimated cost.
- Routes:
  - `/ai/settings`: admin only.
  - `/ai/public-config`: what the public site needs to show the widget, no secrets.
  - `/ai/chat`: public, rate-limited, streamed.
  - `/ai/studio/*`: admin and editor.
  - `/ai/index/*`: admin only.
  - `/blog-posts`: the usual public, admin and ordering routes.
- Rate limits for public chat: per visitor per minute and per day. A daily cap stops all public chat when reached.
- Activity Log entries for AI settings changes, every generated draft, publishing and index rebuilds.

### webnion_frontend (Next.js)

- **Chat widget** on the public site: a round button next to WhatsApp that opens a chat panel. It shows only when the admin
  has it on and placed on the site. Bangla and English, keyboard friendly, works on phones, respects reduced motion.
- **Blog pages**: `/blog` with a list and filters by tag, and `/blog/[slug]` with the article, author, date, related posts,
  SEO tags and share picture. The menu and footer can link to it from Menu & Footer.
- **Admin screens**: AI Settings, AI Studio (chat-style writer), Blog posts (list, edit, publish), AI Usage.
- The Homepage screen gets a Blog section that can be shown or hidden like the others.

## 8. This repository's layout

```
webnion_rag_agent/
  api/index.py                Vercel entry point, exposes the FastAPI app
  app/
    main.py                   FastAPI app, routes, error handling
    config.py                 Service settings from the environment (pydantic-settings)
    security.py               Checks the signature on every request from the Express API
    schemas.py                Request and response models (Pydantic)
    providers/
      chat.py                 Chat models from provider + name + key (init_chat_model)
      images.py               Picture models per provider
      embeddings.py           Embedding models per provider
      capabilities.py         The checks behind the Test button (section 4.2)
      safe_url.py             Public-address check for OpenAI-compatible base addresses
      errors.py               Turns provider errors into plain messages
    rag/
      documents.py            Turns site records into passages with metadata
      store.py                Pinecone upsert, delete and search
    graphs/
      site_assistant.py       Section 5.1
      blog_writer.py          Section 5.2
      prompts.py              Prompts, in English, with the language to answer in
    tracing.py                LangSmith tracing per request, when a key is set
  tests/                      pytest, with fake models and a fake Pinecone
  evals/                      Questions and expected facts for the site assistant
  pyproject.toml, uv.lock     Dependencies, managed with uv
  vercel.json                 Python runtime, Singapore region, five-minute limit
```

Endpoints, all signed by the Express API:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, unsigned |
| `POST /v1/models/test` | Checks a provider, model and key |
| `POST /v1/chat` | Site assistant, streamed |
| `POST /v1/blog/outline` | Blog step 1 |
| `POST /v1/blog/write` | Blog step 2 (text), streamed |
| `POST /v1/images` | One picture from a prompt |
| `POST /v1/index/upsert`, `/v1/index/delete`, `/v1/index/rebuild` | Pinecone updates |

## 9. Security

- **Signed requests.** The Express API signs each request with HMAC-SHA256 over the timestamp and body, using
  `AGENT_SHARED_SECRET`. The AI service rejects unsigned requests, wrong signatures and anything older than five minutes.
- **Keys** are encrypted at rest. They travel to the AI service only inside signed requests over HTTPS, and are never logged.
  LangSmith traces hide the request settings.
- **Public chat**:
  - rate limits and a daily cap,
  - a message length limit,
  - answers limited to Webnion topics,
  - inquiries sent only after the visitor confirms.
- **Base addresses** for OpenAI-compatible providers must be public `https://` addresses (section 4.3).
- **Prompt injection**: site content and visitor text are passed as data in clearly marked blocks. The only action the public
  assistant can take is creating an inquiry, after confirmation.
- **Generated content** is always a draft that a person reviews before it goes live.
- **Costs**: per-day limits for chats, blogs and pictures, with an estimated cost shown in AI Usage.

## 10. Observability and quality

- **LangSmith**: when the admin adds a key, every run is traced with the feature, model and locale as tags. Without a key,
  nothing is sent.
- **Usage table** in MySQL for the admin's AI Usage screen.
- **Evaluation**: about 30 real questions in English and Bangla with the facts a correct answer must contain, such as prices,
  contact details and services. They are run in LangSmith whenever prompts or models change.
- **Tests**:
  - pytest with LangChain's fake chat model and a fake Pinecone, so tests cost nothing,
  - a contract test between the Express API and this service,
  - Playwright tests for the chat widget and AI Studio.

## 11. Environment variables

| Where | Variable | Purpose |
| --- | --- | --- |
| AI service | `AGENT_SHARED_SECRET` | Checks request signatures |
| AI service | `ALLOWED_CLOCK_SKEW_SECONDS` | Optional, default 300 |
| Express API | `AGENT_URL` | Address of this service |
| Express API | `AGENT_SHARED_SECRET` | Signs requests |
| Express API | `AI_SECRETS_KEY` | 32-byte key that encrypts the stored API keys |

Provider, Pinecone and LangSmith keys are **not** environment variables. They are entered in the admin panel.

## 12. Phases

Each phase is pushed, deployed and tested before the next starts.

| Phase | Delivers | Done when |
| --- | --- | --- |
| 0. Skeleton | uv project, FastAPI app, signed requests, `/health`, tests and lint, deployed to Vercel | `/health` answers on Vercel and an unsigned call is refused |
| 1. AI settings | Encrypted settings in the Express API, the AI Settings screen with Test buttons and capability checks, Hugging Face and OpenAI-compatible providers, on/off and placement switches | The owner saves a key and model for each provider type, and Test reports what works or a clear error |
| 2. Search index | Passages from site content, Pinecone upsert, delete and search, Rebuild button, updates on save, nightly check | A search for "ecommerce price" returns the pricing and e-commerce service passages |
| 3. Site assistant | The graph in 5.1, streamed chat, the widget on the site and in admin, inquiries from chat, rate limits | A visitor asks in Bangla and gets a correct Bangla answer with links; a confirmed lead appears under Inquiries |
| 4. Blog section | `BlogPost` model, public blog pages, Blog admin screen, homepage section, menu link | A hand-written post can be published and read in both languages |
| 5. AI Studio | The graph in 5.2, outline approval, pictures, drafts saved to Blog | A topic becomes a reviewed draft with a cover and section pictures in under five minutes |
| 6. Hardening | LangSmith tracing, AI Usage screen, daily caps, evaluation set, documentation | The evaluation passes and the caps stop use when reached |

## 13. What the owner needs to provide

1. **API keys** for at least one chat and writing provider, a picture provider (OpenAI, Google or Hugging Face) and an embedding
   provider. With one OpenAI, Google or Hugging Face key, everything can run on a single provider.
2. **A Pinecone account and API key.** The service creates the index itself.
3. **Optional**: a LangSmith API key for traces and evaluations.
4. **An email service** for inquiry emails, as before. Without it, inquiries from the chat still appear under Inquiries.

## 14. Risks and open points

- **Vercel's five-minute limit** fits one blog post when pictures are made in separate calls. A move to an always-on host would
  be needed only for much longer jobs.
- **Bangla quality** varies by model, and more so among open models on Hugging Face. The evaluation set includes Bangla
  questions so a model change can be checked.
- **Open models may lack tool calling or structured output.** Section 4.2 describes the fallbacks.
- **Picture rights and accuracy**: generated pictures must not imitate client brands. The prompts rule this out, and the
  admin reviews every picture.
- **Costs** depend on the models chosen. The daily caps and the usage screen keep them visible.

## 15. Working rules for this repo

- Python 3.12 or newer, managed with `uv`. Code passes `ruff` and `pytest` before every push.
- Commits use the `Ebratul` identity, with a summary line and a body explaining what and why. **No `Co-Authored-By` or
  "Generated with" lines**: Claude must not appear as a contributor.
