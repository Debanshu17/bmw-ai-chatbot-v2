"""Unit tests for Pillar 3: Cost and Latency Telemetry."""

import pytest
from telemetry.monitor import calculate_cost, TelemetryTracker


@pytest.fixture
def sample_pricing():
    return {
        "gemini-2.5-flash": {
            "input_cost_per_million": 0.075,
            "output_cost_per_million": 0.30
        }
    }


def test_calculate_cost_exact_math(sample_pricing):
    # 10,000 prompt tokens @ $0.075 / 1M = $0.00075
    # 2,000 completion tokens @ $0.30 / 1M = $0.00060
    # Total = 0.00135
    cost = calculate_cost(
        model_name="gemini-2.5-flash",
        prompt_tokens=10000,
        completion_tokens=2000,
        pricing_config=sample_pricing
    )
    assert cost == 0.00135


def test_calculate_cost_zero_tokens(sample_pricing):
    cost = calculate_cost(
        model_name="gemini-2.5-flash",
        prompt_tokens=0,
        completion_tokens=0,
        pricing_config=sample_pricing
    )
    assert cost == 0.0


def test_tracker_measures_duration_and_ttft():
    tracker = TelemetryTracker(model_name="gemini-2.5-flash")
    tracker.start()
    tracker.record_first_token()
    telemetry = tracker.stop(prompt_tokens=500, completion_tokens=150)

    assert telemetry.duration_s >= 0.0
    assert telemetry.ttft_ms >= 0.0
    assert telemetry.prompt_tokens == 500
    assert telemetry.completion_tokens == 150
    assert telemetry.computed_cost_usd > 0.0
