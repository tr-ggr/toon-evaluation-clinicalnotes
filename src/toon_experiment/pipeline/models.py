from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from toon_experiment.config import ModelChoice


@lru_cache(maxsize=8)
def get_chat_model(
    model_choice: ModelChoice, temperature: float, top_p: float, seed: Optional[int]
) -> BaseChatModel:
    """Get ChatOpenAI client configured for OpenRouter endpoint.

    Expects OPENAI_API_KEY to be set (OpenRouter API key).
    Supported models: deepseek-r1-turbo, openai/gpt-4-turbo, anthropic/claude-3.5-sonnet
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (use OpenRouter API key)")

    return ChatOpenAI(
        model=model_choice,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        top_p=top_p,
    )
