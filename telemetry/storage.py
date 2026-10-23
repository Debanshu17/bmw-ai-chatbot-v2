"""Telemetry Persistence and Metrics Aggregation.

Appends execution telemetry to JSONL logs and computes rolling operational statistics
for the /metrics endpoint and evaluation reporting.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from config import settings
from generation.schemas import TelemetryData

logger = logging.getLogger(__name__)


def log_telemetry(
    query: str,
    telemetry: TelemetryData,
    used_fallback: bool,
    citations_count: int,
    log_path: Path = settings.TELEMETRY_LOG_PATH
) -> None:
    """Appends a telemetry record to the JSONL log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "query": query[:100],
        "model": telemetry.model_name,
        "ttft_ms": telemetry.ttft_ms,
        "duration_s": telemetry.duration_s,
        "prompt_tokens": telemetry.prompt_tokens,
        "completion_tokens": telemetry.completion_tokens,
        "cost_usd": telemetry.computed_cost_usd,
        "used_fallback": used_fallback,
        "citations_count": citations_count
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_aggregated_metrics(log_path: Path = settings.TELEMETRY_LOG_PATH) -> Dict[str, Any]:
    """Reads the telemetry log and computes rolling aggregate metrics."""
    if not log_path.exists():
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_ttft_ms": 0.0,
            "total_cost_usd": 0.0,
            "avg_cost_per_query_usd": 0.0,
            "fallback_rate_pct": 0.0
        }

    durations: List[float] = []
    ttfts: List[float] = []
    costs: List[float] = []
    fallbacks: int = 0
    total_records = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                durations.append(data.get("duration_s", 0.0) * 1000.0)
                ttfts.append(data.get("ttft_ms", 0.0))
                costs.append(data.get("cost_usd", 0.0))
                if data.get("used_fallback", False):
                    fallbacks += 1
                total_records += 1
            except Exception as e:
                logger.error(f"Error parsing telemetry log entry: {e}")

    if total_records == 0:
        return {"total_queries": 0}

    dur_arr = np.array(durations)
    return {
        "total_queries": total_records,
        "avg_latency_ms": round(float(np.mean(dur_arr)), 2),
        "p50_latency_ms": round(float(np.percentile(dur_arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(dur_arr, 95)), 2),
        "avg_ttft_ms": round(float(np.mean(ttfts)), 2) if ttfts else 0.0,
        "total_cost_usd": round(float(np.sum(costs)), 6),
        "avg_cost_per_query_usd": round(float(np.mean(costs)), 6) if costs else 0.0,
        "fallback_rate_pct": round((fallbacks / total_records) * 100.0, 2)
    }
