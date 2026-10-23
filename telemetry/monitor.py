"""Cost and Latency Telemetry Monitor.

Measures Time-To-First-Token (TTFT), total latency, prompt/completion token usage,
and computes dynamic financial cost derived from config/pricing.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from config import settings
from generation.schemas import TelemetryData

logger = logging.getLogger(__name__)


def load_pricing_config(config_path: Path = settings.PRICING_CONFIG_PATH) -> Dict[str, Any]:
    """Loads model pricing table from JSON config."""
    if not config_path.exists():
        logger.warning(f"Pricing config not found at {config_path}. Using zero cost fallback.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_config: Optional[Dict[str, Any]] = None
) -> float:
    """Calculates query cost in USD using per-1M-token pricing rates.

    Args:
        model_name: Target LLM model string (e.g. 'gemini-2.5-flash').
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of generated output tokens.
        pricing_config: Optional pricing dictionary.

    Returns:
        Computed cost in USD rounded to 6 decimal places.
    """
    if pricing_config is None:
        pricing_config = load_pricing_config()

    # Normalize model key
    clean_model = model_name.replace("models/", "")
    rates = pricing_config.get(clean_model) or pricing_config.get(model_name) or {}

    input_rate = rates.get("input_cost_per_million", 0.075)
    output_rate = rates.get("output_cost_per_million", 0.30)

    input_cost = (prompt_tokens / 1_000_000.0) * input_rate
    output_cost = (completion_tokens / 1_000_000.0) * output_rate
    total_cost = round(input_cost + output_cost, 6)
    return total_cost


class TelemetryTracker:
    """Execution timer and token tracker for LLM and RAG pipelines."""

    def __init__(self, model_name: str = settings.CHAT_MODEL):
        self.model_name = model_name
        self.start_time: float = 0.0
        self.first_token_time: Optional[float] = None
        self.end_time: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def start(self) -> "TelemetryTracker":
        """Starts timing."""
        self.start_time = time.perf_counter()
        return self

    def record_first_token(self) -> None:
        """Records timestamp of first streamed token."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def stop(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> TelemetryData:
        """Stops timing, aggregates metrics, and returns TelemetryData."""
        self.end_time = time.perf_counter()
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

        total_duration = round(self.end_time - self.start_time, 4)

        if self.first_token_time is not None:
            ttft_ms = round((self.first_token_time - self.start_time) * 1000.0, 2)
        else:
            ttft_ms = round(total_duration * 1000.0, 2)

        cost = calculate_cost(
            model_name=self.model_name,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens
        )

        return TelemetryData(
            ttft_ms=ttft_ms,
            duration_s=total_duration,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            model_name=self.model_name,
            computed_cost_usd=cost
        )
