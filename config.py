"""Application configuration and environment settings.

Manages LLM provider configuration, validation limits, timeouts, and application metadata
using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for AI Hallucination Hunter."""

    # Application Metadata
    app_name: str = "AI Hallucination Hunter"
    app_version: str = "1.0.0"
    environment: str = "development"

    # LLM Provider Configuration
    # Supports OpenAI or any OpenAI-compatible provider (e.g. Groq, Ollama, Together, DeepSeek, LocalAI)
    llm_provider: str = Field(default="openai", description="Provider identifier (e.g., openai, groq, ollama)")
    llm_api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    llm_base_url: Optional[str] = Field(default=None, description="Base URL for OpenAI-compatible endpoint")
    llm_model: str = Field(default="gpt-4o-mini", description="Model name to use for claim extraction and evaluation")
    llm_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0, description="Timeout for LLM API calls in seconds")
    llm_max_retries: int = Field(default=2, ge=0, le=5, description="Number of retry attempts on transient LLM errors")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature for factual verification (0.0 for deterministic output)")

    # Input Size & Resource Limits (Denial-of-Service Defense)
    max_candidate_answer_chars: int = Field(default=5000, ge=10, le=50000, description="Maximum characters for candidate answer")
    max_evidence_items: int = Field(default=20, ge=1, le=100, description="Maximum number of evidence items per request")
    max_evidence_text_chars: int = Field(default=10000, ge=10, le=50000, description="Maximum characters per individual evidence item")
    max_total_chars: int = Field(default=50000, ge=100, le=200000, description="Maximum total characters across candidate answer and all evidence")
    max_extracted_claims: int = Field(default=30, ge=1, le=100, description="Maximum claims to extract and verify from an answer")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
