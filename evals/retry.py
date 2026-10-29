"""API Retry Logic and Infrastructure Failure Detection.

Provides exponential backoff retries for transient Google Gemini API errors (429, timeout, 503)
and distinguishes between legitimate model outputs and infrastructure failures.
"""

import logging
import re
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


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
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 32.0
) -> Tuple[Optional[Any], bool, Optional[str]]:
    """Executes a Gemini API call with exponential backoff on transient infrastructure failures.

    Args:
        fn: Zero-argument callable invoking the API.
        max_retries: Maximum number of retry attempts (default: 5).
        initial_backoff: Initial backoff delay in seconds (default: 1.0).
        backoff_factor: Exponential multiplier (default: 2.0).
        max_backoff: Maximum backoff delay cap (default: 32.0).

    Returns:
        Tuple of (result, is_error: bool, error_message: Optional[str]).
        - If call succeeds: (result, False, None)
        - If transient API failure persists after max_retries: (None, True, error_msg)
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            res = fn()
            return res, False, None
        except Exception as e:
            last_exc = e
            if is_api_failure(e):
                if attempt < max_retries:
                    sleep_time = min(initial_backoff * (backoff_factor ** attempt), max_backoff)
                    # If API explicitly suggests a retry delay (e.g. "retry in 18s"), respect it
                    delay_match = re.search(r"retry(?:\s+in|\s+after)?\s+([\d\.]+)\s*s", str(e), re.IGNORECASE)
                    if delay_match:
                        suggested_delay = float(delay_match.group(1)) + 1.0
                        sleep_time = max(sleep_time, min(suggested_delay, max_backoff))

                    logger.warning(
                        f"Transient API failure ({type(e).__name__}: {e}). "
                        f"Retrying with exponential backoff in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})..."
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
