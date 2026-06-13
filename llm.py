"""Single-provider model factory.

Everything (all four agents + the image-PDF vision OCR) runs through OpenRouter, so
the user only ever needs ONE key: OPENROUTER_API_KEY. OpenRouter is OpenAI-compatible,
so we talk to it with `ChatOpenAI` pointed at its base URL. Swap the model strings or
the client here to repoint the whole app at a different provider/model.
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# One model powers the whole app. Change this single line to swap it — any OpenRouter
# model id works (e.g. "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"). Structured
# output (the synthesis report) needs a tool-calling model.
MODEL = "google/gemini-2.5-flash"

# Used to transcribe image-based PDFs (pitch decks). Must be a multimodal model; the
# default chat model already is, so we reuse it.
VISION_MODEL = MODEL


def get_llm(model: str, api_key: str, temperature: float = 0.3) -> ChatOpenAI:
    """Chat LLM used by every agent (and the vision OCR), via OpenRouter."""
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        # Optional but recommended by OpenRouter for attribution / rankings.
        default_headers={
            "HTTP-Referer": "https://github.com/deep-research-agent",
            "X-Title": "Deep Research Agent",
        },
    )


def set_global_key(api_key: str) -> None:
    """Mirror the key into the environment for any SDK that reads it from there."""
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
