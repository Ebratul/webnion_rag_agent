"""Embedding models for the Pinecone search index."""

from langchain_core.embeddings import Embeddings

from app.providers.errors import ProviderError
from app.providers.fake import HashingEmbeddings, require_fake_allowed
from app.providers.safe_url import ensure_public_url
from app.schemas import ModelConfig

FAKE_DIMENSION = 384


async def build_embeddings(config: ModelConfig, *, timeout: float = 60) -> Embeddings:
    key = config.api_key.get_secret_value()

    if config.provider == "fake":
        require_fake_allowed()
        return HashingEmbeddings(FAKE_DIMENSION)

    if config.provider in ("openai", "openai_compatible"):
        from langchain_openai import OpenAIEmbeddings

        base_url = await ensure_public_url(config.base_url or "") if config.provider == "openai_compatible" else None
        # The length check downloads a tokenizer at run time and does not know other providers' models. Passages are kept
        # well under every model's limit instead.
        return OpenAIEmbeddings(model=config.model, api_key=key, base_url=base_url, timeout=timeout, max_retries=1, check_embedding_ctx_length=False)

    if config.provider == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=config.model, google_api_key=key)

    if config.provider == "mistralai":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=config.model, api_key=key, timeout=int(timeout), max_retries=1)

    if config.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        if config.base_url:
            return HuggingFaceEndpointEmbeddings(model=await ensure_public_url(config.base_url), huggingfacehub_api_token=key)
        return HuggingFaceEndpointEmbeddings(model=config.model, provider=config.hf_provider or "auto", huggingfacehub_api_token=key)

    raise ProviderError("not_supported", f"{config.provider} has no embedding models here")
