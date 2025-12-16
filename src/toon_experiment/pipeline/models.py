from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from toon_experiment.config import ModelChoice


@lru_cache(maxsize=8)
def get_chat_model(
    model_choice: ModelChoice, temperature: float, top_p: float, seed: Optional[int]
) -> BaseChatModel:
    """Get Gemini chat model client.

    Expects GOOGLE_API_KEY to be set.
    Supported models: gemini-2.5-pro
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    return ChatGoogleGenerativeAI(
        model=model_choice,
        google_api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
