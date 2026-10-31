"""API Retry Logic and Infrastructure Failure Detection.

Provides exponential backoff retries for transient Google Gemini API errors (429, timeout, 503)
and distinguishes between legitimate model outputs and infrastructure failures.
"""

import logging
import re
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# Tracks last API request timestamp to enforce max 8 requests per minute (7.5s interval)
_last_request_time: float = 0.0


def pace_api_request(min_interval: float = 7.5) -> None:
    """Enforces request pacing to cap Gemini API calls at 8 requests per minute."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < min_interval:
        sleep_duration = min_interval - elapsed
        time.sleep(sleep_duration)
    _last_request_time = time.time()


def is_api_failure(exc: Exception) -> bool:
    """Detects whether an exception represents a transient API infrastructure error.

    Covers rate limits (429, quota exhaustion), timeouts (deadline exceeded), and 503 errors.
    """
    exc_type = type(exc).__name__.lower()
    msg = str(exc).lower()

    # Check exception type names
    transient_types = [
        "resourceexhausted",
        "deadlineexceeded",
        "serviceunavailable",
        "timeout",
        "toomanyrequests",
        "internalservererror"
    ]
    if any(k in exc_type for k in transient_types):
        return True

    # Check error message strings
    api_indicators = [
        "429", "quota", "rate limit", "resource_exhausted",
        "deadline exceeded", "timed out", "timeout",
        "503", "service unavailable", "500", "internal server error",
        "connection reset", "connection refused", "remotedisconnected",
        "overloaded"
    ]
    return any(ind in msg for ind in api_indicators)


def call_gemini_with_retry(
    fn: Callable[[], Any],
    max_retries: int = 8,
    initial_backoff: float = 2.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 65.0,
    min_request_interval: float = 7.5
) -> Tuple[Optional[Any], bool, Optional[str]]:
    """Executes a Gemini API call with rate pacing (max 8 req/min) and stop-and-wait retry on 429.

    Args:
        fn: Zero-argument callable invoking the API.
        max_retries: Maximum number of retry attempts (default: 8).
        initial_backoff: Initial backoff delay in seconds (default: 2.0).
        backoff_factor: Exponential multiplier (default: 2.0).
        max_backoff: Maximum backoff delay cap (default: 65.0).
        min_request_interval: Minimum seconds between requests (default: 7.5s = 8 req/min).

    Returns:
        Tuple of (result, is_error: bool, error_message: Optional[str]).
        - If call succeeds: (result, False, None)
        - If transient API failure persists after max_retries: (None, True, error_msg)
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        # Enforce rate pacing (8 requests per minute) unless running fast unit tests
        if initial_backoff >= 0.1:
            pace_api_request(min_interval=min_request_interval)

        try:
            res = fn()
            return res, False, None
        except Exception as e:
            last_exc = e
            if is_api_failure(e):
                if attempt < max_retries:
                    base_sleep = min(initial_backoff * (backoff_factor ** attempt), max_backoff)
                    err_str = str(e).lower()

                    # Treat 429 as stop-and-wait: check if API specifies delay or wait for quota reset
                    delay_match = re.search(r"retry(?:\s+in|\s+after)?\s+([\d\.]+)\s*s", str(e), re.IGNORECASE)
                    if delay_match:
                        suggested_delay = float(delay_match.group(1)) + 1.0
                        sleep_time = max(base_sleep, min(suggested_delay, max_backoff))
                    elif "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                        # Stop and wait on 429: wait at least 20s (or 0.05s in test mode)
                        sleep_time = max(base_sleep, 20.0 if initial_backoff >= 0.1 else 0.05)
                        sleep_time = min(sleep_time, max_backoff)
                    else:
                        sleep_time = base_sleep

                    logger.warning(
                        f"Transient API failure ({type(e).__name__}: {e}). "
                        f"Stop-and-wait backoff for {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    err_msg = f"API failure persisted after {max_retries} retries: {type(e).__name__}: {e}"
                    logger.error(err_msg)
                    return None, True, err_msg
            else:
                # Non-transient exception (e.g. invalid arguments, configuration error)
                logger.error(f"Non-transient error encountered during API call: {type(e).__name__}: {e}")
                raise e

    return None, True, str(last_exc)
