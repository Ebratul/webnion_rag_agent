"""Chat models built from the provider, model name and key the admin typed."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.providers.errors import ProviderError
from app.providers.fake import FakeChatModel, require_fake_allowed
from app.providers.safe_url import ensure_public_url
from app.schemas import ModelConfig

# Providers that init_chat_model builds directly
_BUILTIN = {"openai", "anthropic", "google_genai", "groq", "mistralai"}


async def build_chat_model(config: ModelConfig, *, max_tokens: int = 1024, timeout: float = 90) -> BaseChatModel:
    """Builds the client. Nothing is sent to the provider until the model is called.

    temperature, top_p and top_k are never set: current Claude models and OpenAI's reasoning models reject them, and every
    provider's default suits these jobs.
    """
    key = config.api_key.get_secret_value()

    if config.provider == "fake":
        require_fake_allowed()
        return FakeChatModel()

    if config.provider in _BUILTIN:
        extra: dict[str, object] = {"stream_usage": True} if config.provider == "openai" else {}
        return init_chat_model(config.model, model_provider=config.provider, api_key=key, max_tokens=max_tokens, timeout=timeout, max_retries=1, **extra)

    if config.provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        base_url = await ensure_public_url(config.base_url or "")
        return ChatOpenAI(model=config.model, api_key=key, base_url=base_url, max_tokens=max_tokens, timeout=timeout, max_retries=1, stream_usage=True)

    if config.provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        if config.base_url:
            # A dedicated Inference Endpoint the admin pasted
            endpoint = HuggingFaceEndpoint(
                endpoint_url=await ensure_public_url(config.base_url),
                huggingfacehub_api_token=key,
                max_new_tokens=max_tokens,
                timeout=timeout,
                task="text-generation",
            )
        else:
            endpoint = HuggingFaceEndpoint(
                repo_id=config.model,
                provider=config.hf_provider or "auto",
                huggingfacehub_api_token=key,
                max_new_tokens=max_tokens,
                timeout=timeout,
                task="text-generation",
            )
        return ChatHuggingFace(llm=endpoint, model_id=config.model)

    raise ProviderError("not_supported", f"unknown provider {config.provider}")


def text_of(message: BaseMessage) -> str:
    """Plain text of a reply. Some providers return a list of blocks, including thinking, rather than a string."""
    return message.text if isinstance(message.text, str) else str(message.text())
