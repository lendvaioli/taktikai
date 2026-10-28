"""Configuration models of the report stage (configs/report.yaml)."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

BackendName = Literal["ollama", "anthropic"]


class OllamaConfig(BaseModel):
    """Local open model served by Ollama (default backend, free)."""

    host: str = "http://127.0.0.1:11434"
    model: str = "hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"
    think: bool = False
    timeout_sec: float = Field(900.0, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)


class AnthropicConfig(BaseModel):
    """Anthropic API backend (optional; needs ANTHROPIC_API_KEY or another SDK credential)."""

    model: str = "claude-opus-5-5"
    max_tokens: int = Field(16000, gt=0)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"


class ReportConfig(BaseModel):
    """Contents of configs/report.yaml."""

    backend: BackendName = "ollama"
    backend_env: str = "TAKTIKAI_LLM_BACKEND"
    prompt_path: Path = Path("prompts/report_system.md")
    max_steps: int = Field(8, ge=1)
    retries: int = Field(1, ge=0)
    number_tolerance: float = Field(0.5, ge=0)
    max_events: int = Field(30, ge=1)
    ollama: OllamaConfig = OllamaConfig()
    anthropic: AnthropicConfig = AnthropicConfig()
