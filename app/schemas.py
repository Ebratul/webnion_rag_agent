"""Request and response models. Everything the Express API sends is validated here before it is used."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, field_validator, model_validator

Locale = Literal["en", "bn"]
ChatProvider = Literal["openai", "anthropic", "google_genai", "groq", "mistralai", "huggingface", "openai_compatible", "fake"]
EmbeddingProvider = Literal["openai", "google_genai", "mistralai", "huggingface", "openai_compatible", "fake"]
ImageProvider = Literal["openai", "google_genai", "huggingface", "cloudflare", "fake"]

ModelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/@+-]+$")]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{1,45}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(Strict):
    """One provider, model name and key, as the admin typed them."""

    provider: str
    model: ModelName
    api_key: SecretStr = Field(default=SecretStr(""), max_length=4096)
    # OpenAI-compatible providers, and Hugging Face dedicated endpoints
    base_url: str | None = Field(default=None, max_length=500)
    # Hugging Face only: which inference provider serves the model, such as "auto"
    hf_provider: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{1,40}$")] | None = None
    # Cloudflare Workers AI only: the account the models run in
    account_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")] | None = None

    @model_validator(mode="after")
    def _key_needed(self) -> "ModelConfig":
        if self.provider != "fake" and not self.api_key.get_secret_value().strip():
            raise ValueError("An API key is required")
        if self.provider == "cloudflare" and not self.account_id:
            raise ValueError("Cloudflare needs the account ID")
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("An OpenAI-compatible provider needs a base address")
        return self


class ChatModelConfig(ModelConfig):
    provider: ChatProvider
    # From the last Test. Without tool calling the assistant offers the form instead of taking an inquiry in the chat.
    tools: bool = True
    structured: bool = True


class EmbeddingConfig(ModelConfig):
    provider: EmbeddingProvider


class ImageModelConfig(ModelConfig):
    provider: ImageProvider


class PineconeConfig(Strict):
    api_key: SecretStr = Field(max_length=500)
    index: Slug
    cloud: Literal["aws", "gcp", "azure"] = "aws"
    region: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{3,30}$")] = "us-east-1"


class LangSmithConfig(Strict):
    api_key: SecretStr = Field(max_length=500)
    project: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)] = "webnion"


class Document(Strict):
    """One record of site content in one language."""

    type: Annotated[str, StringConstraints(pattern=r"^[a-z-]{1,40}$")]
    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,100}$")]
    locale: Locale
    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]
    url: Annotated[str, StringConstraints(max_length=500)] = ""
    text: Annotated[str, StringConstraints(max_length=60_000)]
    updated_at: Annotated[str, StringConstraints(max_length=40)] = ""

    @field_validator("url")
    @classmethod
    def _site_link(cls, value: str) -> str:
        # Shown to visitors as a link, so only site paths and https addresses are kept
        if value and not (value.startswith("/") and not value.startswith("//")) and not value.startswith("https://"):
            return ""
        return value


class Retrieval(Strict):
    """Where the assistant looks for site content: Pinecone when it is set up, otherwise the records sent with the request."""

    embedding: EmbeddingConfig | None = None
    pinecone: PineconeConfig | None = None
    documents: list[Document] = Field(default_factory=list, max_length=600)


class Contact(Strict):
    email: ShortText = ""
    phone: ShortText = ""
    whatsapp: ShortText = ""
    contact_url: ShortText = ""


class AssistantSettings(Strict):
    site_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)] = "Webnion"
    name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=60)] = ""
    tone: Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)] = ""
    allow_inquiry: bool = True
    max_output_tokens: int = Field(default=700, ge=64, le=2000)
    contact: Contact = Field(default_factory=Contact)


class ChatTurn(Strict):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(max_length=4000)]


class ChatRequest(Strict):
    locale: Locale = "en"
    audience: Literal["public", "admin"] = "public"
    messages: list[ChatTurn] = Field(min_length=1, max_length=20)
    chat_model: ChatModelConfig
    retrieval: Retrieval = Field(default_factory=Retrieval)
    assistant: AssistantSettings = Field(default_factory=AssistantSettings)
    langsmith: LangSmithConfig | None = None

    @model_validator(mode="after")
    def _ends_with_visitor(self) -> "ChatRequest":
        last = self.messages[-1]
        if last.role != "user":
            raise ValueError("The last message must be the visitor's")
        if len(last.content) > 1500:
            raise ValueError("Message is too long")
        return self


class ModelTestRequest(Strict):
    kind: Literal["chat", "embedding", "image"]
    model: ModelConfig
    # Making a test picture costs money, so it runs only when asked
    make_picture: bool = False
    langsmith: LangSmithConfig | None = None

    @model_validator(mode="after")
    def _provider_fits(self) -> "ModelTestRequest":
        allowed = {"chat": ChatProvider, "embedding": EmbeddingProvider, "image": ImageProvider}[self.kind]
        if self.model.provider not in allowed.__args__:
            raise ValueError(f"{self.model.provider} cannot be used for {self.kind}")
        return self


class ModelTestResult(BaseModel):
    ok: bool
    answers: bool = False
    tools: bool | None = None
    structured: bool | None = None
    dimension: int | None = None
    picture: bool | None = None
    latency_ms: int = 0
    error: dict[str, str] | None = None


class OutlineSection(BaseModel):
    heading: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    points: list[Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)]] = Field(default_factory=list, max_length=8)


class Outline(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    angle: Annotated[str, StringConstraints(strip_whitespace=True, max_length=600)] = ""
    sections: list[OutlineSection] = Field(min_length=2, max_length=10)


class BlogBrief(Strict):
    topic: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=300)]
    audience: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""
    tone: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] = ""
    length: Literal["short", "medium", "long"] = "medium"
    keywords: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]] = Field(default_factory=list, max_length=10)
    notes: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1500)] = ""


class OutlineRequest(Strict):
    brief: BlogBrief
    writing_model: ChatModelConfig
    retrieval: Retrieval = Field(default_factory=Retrieval)
    site_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)] = "Webnion"
    langsmith: LangSmithConfig | None = None


class WriteRequest(OutlineRequest):
    outline: Outline
    languages: list[Locale] = Field(default_factory=lambda: ["en"], min_length=1, max_length=2)
    max_pictures: int = Field(default=3, ge=0, le=6)
    picture_style: Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)] = ""


class ImageRequest(Strict):
    image_model: ImageModelConfig
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]
    aspect: Literal["landscape", "square"] = "landscape"
    langsmith: LangSmithConfig | None = None


class IndexUpsertRequest(Strict):
    embedding: EmbeddingConfig
    pinecone: PineconeConfig
    documents: list[Document] = Field(default_factory=list, max_length=600)
    # Rebuild: empties the index first, then writes everything sent
    replace_all: bool = False


class IndexItem(Strict):
    type: Annotated[str, StringConstraints(pattern=r"^[a-z-]{1,40}$")]
    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,100}$")]


class IndexDeleteRequest(Strict):
    pinecone: PineconeConfig
    items: list[IndexItem] = Field(min_length=1, max_length=200)


class SearchRequest(Strict):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    locale: Locale = "en"
    retrieval: Retrieval


class TextBlock(BaseModel):
    type: Literal["heading", "paragraph"]
    text: Annotated[str, StringConstraints(max_length=5000)]


class ListBlock(BaseModel):
    type: Literal["list"]
    items: list[Annotated[str, StringConstraints(max_length=5000)]] = Field(max_length=50)


class ImageBlock(BaseModel):
    type: Literal["image"]
    src: Annotated[str, StringConstraints(max_length=2048)]
    alt: Annotated[str, StringConstraints(max_length=300)] = ""


Block = Annotated[TextBlock | ListBlock | ImageBlock, Field(discriminator="type")]


class PostDraft(BaseModel):
    title: Annotated[str, StringConstraints(max_length=200)]
    excerpt: Annotated[str, StringConstraints(max_length=400)] = ""
    blocks: list[Block] = Field(max_length=300)
    seo_title: Annotated[str, StringConstraints(max_length=200)] = ""
    seo_description: Annotated[str, StringConstraints(max_length=400)] = ""
    tags: list[Annotated[str, StringConstraints(max_length=60)]] = Field(default_factory=list, max_length=10)


class TranslateRequest(Strict):
    writing_model: ChatModelConfig
    post: PostDraft
    target: Literal["bn"] = "bn"
    langsmith: LangSmithConfig | None = None
