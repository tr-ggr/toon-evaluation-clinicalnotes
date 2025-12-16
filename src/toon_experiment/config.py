from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

# Load .env early for CLI usage
load_dotenv(override=False)


OutputFormat = Literal["json", "yaml", "toon"]
ModelChoice = Literal["deepseek-r1-turbo", "openai/gpt-4-turbo", "anthropic/claude-3.5-sonnet"]


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("data"))
    outputs_dir: Path = Field(default=Path("outputs"))
    format: OutputFormat = Field(default="json")
    model: ModelChoice = Field(default="deepseek-chimera")
    max_retries: int = Field(default=3)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.9)
    seed: Optional[int] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACEHUB_API_TOKEN")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("max_retries")
    def _max_retries_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v


settings = Settings()
