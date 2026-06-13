"""Single-provider model factory.

Everything (all four agents + the RAG embeddings) runs on Google Gemini, so the
user only ever needs ONE key: GOOGLE_API_KEY. Swap the implementations here if you
ever want to point the whole app at a different provider.
"""
from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

# One model powers the whole app. Change this single line to swap it.
# (gemini-3.1-pro-preview also works — it's newer but slower and in preview.)
MODEL = "gemini-2.5-pro"
EMBED_MODEL = "gemini-embedding-001"


def get_llm(model: str, api_key: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Chat LLM used by every agent."""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )


def get_embed_model(api_key: str):
    """Embedding model used by the RAG agent's in-memory vector index."""
    # Imported lazily so the app still starts even if this optional path has issues.
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    return GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=api_key)


def set_global_key(api_key: str) -> None:
    """Some SDKs read the key from the environment; mirror it there to be safe.

    Only GOOGLE_API_KEY is set — the google-genai SDK warns if both GOOGLE_API_KEY
    and GEMINI_API_KEY are present."""
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ.pop("GEMINI_API_KEY", None)
