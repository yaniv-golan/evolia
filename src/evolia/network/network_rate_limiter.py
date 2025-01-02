"""Rate limiting functionality for network requests"""
import time
import logging
from typing import List, Optional
from functools import wraps

from .logger import setup_logger

logger = setup_logger()


class RateLimiter:
    """Rate limiter for network requests"""

    def __init__(self, max_calls: int = 100, period: int = 60):
        """Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.call_times: List[float] = []

    def _cleanup_old_calls(self):
        """Remove calls older than the period"""
        current_time = time.time()
        self.call_times = [t for t in self.call_times if current_time - t < self.period]

    def check_rate_limit(self) -> bool:
        """Check if the rate limit has been exceeded

        Returns:
            bool: True if rate limit is not exceeded, False otherwise
        """
        self._cleanup_old_calls()
        return len(self.call_times) < self.max_calls

    def record_call(self):
        """Record a new call"""
        self.call_times.append(time.time())

    def __call__(self, func):
        """Decorator to apply rate limiting to a function

        Args:
            func: Function to decorate

        Returns:
            Wrapped function with rate limiting
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not self.check_rate_limit():
                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "payload": {
                            "max_calls": self.max_calls,
                            "period": self.period,
                            "current_calls": len(self.call_times),
                        }
                    },
                )
                raise Exception(
                    f"Rate limit exceeded: {self.max_calls} calls per {self.period} seconds"
                )

            self.record_call()
            return func(*args, **kwargs)

        return wrapped
