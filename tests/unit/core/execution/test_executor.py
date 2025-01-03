"""Tests for executor functionality."""
import pytest

from evolia.core.executor2 import Executor2 as Executor
from evolia.utils.exceptions import ExecutorError


def test_executor_initialization(is_github_actions):
    """Test executor initialization."""
    if is_github_actions:
        pytest.skip("Skipping OpenAI-dependent test in GitHub Actions")

    config = {
        "allowed_modules": ["math", "typing"],
        "max_runtime": 10,
        "openai": {
            "api_key": "test-key",  # Mock API key for testing
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.7,
        },
    }
    executor = Executor(config)
    assert executor.config == config
