"""Tests for executor functionality."""
import pytest

from evolia.core.executor2 import Executor2 as Executor
from evolia.utils.exceptions import ExecutorError


def test_executor_initialization():
    """Test executor initialization."""
    config = {
        "allowed_modules": ["math", "typing"],
        "max_runtime": 10,
    }
    executor = Executor(config)
    assert executor.config == config
