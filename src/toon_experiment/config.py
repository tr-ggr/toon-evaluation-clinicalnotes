from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env early for CLI usage
load_dotenv(override=False)


OutputFormat = Literal["json", "yaml", "toon"]
ModelChoice = Literal["gemini-2.5-pro"]


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("data"))
    outputs_dir: Path = Field(default=Path("outputs"))
    format: OutputFormat = Field(default="json")
    model: ModelChoice = Field(default="gemini-2.5-pro")
    max_retries: int = Field(default=3)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.9)
    seed: Optional[int] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    huggingface_token: Optional[str] = Field(default=None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @field_validator("seed", mode="before")
    @classmethod
    def _seed_optional(cls, v: object) -> Optional[int]:
        """Convert empty string to None; parse non-empty strings as int."""
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        if isinstance(v, str):
            return int(v)
        return v  # type: ignore[return-value]

    @field_validator("max_retries")
    @classmethod
    def _max_retries_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v


settings = Settings()
