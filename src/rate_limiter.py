"""Rate limiting utilities for API calls with exponential backoff and retry logic."""

import time
import logging
from typing import Dict, Optional, Callable, Any
from functools import wraps
from dataclasses import dataclass, field
import random
import threading

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


@dataclass
class RateLimitState:
    """State tracking for rate limiting."""

    last_request_time: float = field(default_factory=time.time)
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    minute_start: float = field(default_factory=time.time)
    hour_start: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RateLimiter:
    """Thread-safe rate limiter with multiple time window constraints."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.states: Dict[str, RateLimitState] = {}

    def _get_state(self, key: str = "default") -> RateLimitState:
        """Get or create rate limit state for a key."""
        if key not in self.states:
            self.states[key] = RateLimitState()
        return self.states[key]

    def _reset_counters_if_needed(self, state: RateLimitState, now: float):
        """Reset counters when time windows expire."""
        # Reset minute counter
        if now - state.minute_start >= 60:
            state.requests_this_minute = 0
            state.minute_start = now

        # Reset hour counter
        if now - state.hour_start >= 3600:
            state.requests_this_hour = 0
            state.hour_start = now

    def _calculate_delay(self, state: RateLimitState, now: float) -> float:
        """Calculate required delay before next request."""
        delays = []

        # Per-second rate limiting
        time_since_last = now - state.last_request_time
        if time_since_last < (1.0 / self.config.requests_per_second):
            delays.append((1.0 / self.config.requests_per_second) - time_since_last)

        # Per-minute rate limiting
        if state.requests_this_minute >= self.config.requests_per_minute:
            time_until_minute_reset = 60 - (now - state.minute_start)
            if time_until_minute_reset > 0:
                delays.append(time_until_minute_reset)

        # Per-hour rate limiting
        if state.requests_this_hour >= self.config.requests_per_hour:
            time_until_hour_reset = 3600 - (now - state.hour_start)
            if time_until_hour_reset > 0:
                delays.append(time_until_hour_reset)

        return max(delays) if delays else 0

    def wait_if_needed(self, key: str = "default"):
        """Wait if rate limit would be exceeded."""
        state = self._get_state(key)

        with state.lock:
            now = time.time()
            self._reset_counters_if_needed(state, now)

            delay = self._calculate_delay(state, now)
            if delay > 0:
                logger.info(f"Rate limit reached for {key}, waiting {delay:.2f}s")
                time.sleep(delay)
                now = time.time()

            # Update state
            state.last_request_time = now
            state.requests_this_minute += 1
            state.requests_this_hour += 1

    def get_status(self, key: str = "default") -> Dict[str, Any]:
        """Get current rate limiting status."""
        state = self._get_state(key)
        now = time.time()

        with state.lock:
            self._reset_counters_if_needed(state, now)

            return {
                "requests_this_minute": state.requests_this_minute,
                "requests_this_hour": state.requests_this_hour,
                "time_until_minute_reset": max(0, 60 - (now - state.minute_start)),
                "time_until_hour_reset": max(0, 3600 - (now - state.hour_start)),
                "time_since_last_request": now - state.last_request_time,
            }


def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
):
    """Decorator for exponential backoff retry logic."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def rate_limited(config: Optional[RateLimitConfig] = None, key: Optional[str] = None):
    """Decorator for rate limiting function calls."""

    if config is None:
        config = RateLimitConfig()

    # Global rate limiter instance
    if not hasattr(rate_limited, "_limiter"):
        rate_limited._limiter = RateLimiter(config)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter_key = key or func.__name__
            rate_limited._limiter.wait_if_needed(limiter_key)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-configured rate limiters for common APIs
BRAVE_SEARCH_CONFIG = RateLimitConfig(
    requests_per_second=0.5,  # Conservative: 30 requests/minute
    requests_per_minute=30,
    requests_per_hour=1800,  # Conservative for free tier
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
)

OPENAI_CONFIG = RateLimitConfig(
    requests_per_second=0.33,  # 20 requests/minute for GPT-4
    requests_per_minute=20,
    requests_per_hour=1000,  # Generous for typical usage
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
)

WEB_SCRAPING_CONFIG = RateLimitConfig(
    requests_per_second=2.0,  # Respectful web scraping
    requests_per_minute=120,
    requests_per_hour=7200,
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
)

# Global rate limiter instances
brave_limiter = RateLimiter(BRAVE_SEARCH_CONFIG)
openai_limiter = RateLimiter(OPENAI_CONFIG)
web_limiter = RateLimiter(WEB_SCRAPING_CONFIG)


def get_rate_limiter_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all rate limiters."""
    return {
        "brave_search": brave_limiter.get_status("brave_search"),
        "openai": openai_limiter.get_status("openai"),
        "web_scraping": web_limiter.get_status("web_scraping"),
    }
