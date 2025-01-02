"""Tests for Executor class."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import logging
import json

from evolia.core.executor import Executor
from evolia.models import Plan, PlanStep, Parameter, OutputDefinition
from evolia.utils.exceptions import ExecutorError

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        "openai": {
            "model": "gpt-4o-2024-08-06",
            "api_key": "test-key",
            "temperature": 0.7,
        },
        "python_generation": {
            "prompt_template": """
Write a Python function with these requirements:

Description: {description}

Function Name: {function_name}
Parameters:
{parameters}
Return Type: {return_type}
Constraints:
{constraints}

Example Format:
{example_format}

Return only the function definition, no explanations.
"""
        },
    }


@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI API calls."""
    with patch("evolia.integrations.openai_structured.OpenAI") as mock:
        mock_client = MagicMock()
        mock_response = {
            "code": "def test_function(x: int) -> int:\n    return abs(x)",
            "function_info": {
                "name": "test_function",
                "parameters": [{"name": "x", "type": "int"}],
                "return_type": "int",
                "docstring": "Return the absolute value of the input number x.",
            },
            "validation_results": {"syntax_valid": True, "security_issues": []},
            "outputs": {},
            "required_imports": [],
        }
        mock_client.chat.completions.create.return_value = MagicMock(
            id="test-id",
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock.return_value = mock_client
        yield mock


@pytest.fixture
def executor(mock_config):
    """Create a test executor."""
    return Executor(config=mock_config)


def test_execute_plan(executor):
    """Test basic plan execution."""
    step = PlanStep(
        name="test_step",
        tool="generate_code",
        inputs={
            "function_name": "test_function",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "description": "Return the square of the input number x. For example, if x is 5, return 25.",
        },
        outputs={"result": OutputDefinition(type="str")},
    )

    plan = Plan(steps=[step], artifacts_dir="test_artifacts")

    result = executor.execute_plan(plan)
    assert result is not None
    assert len(result) > 0


def test_error_handling(executor):
    """Test error handling during execution."""
    step = PlanStep(
        name="test_step",
        tool="generate_code",
        inputs={
            "function_name": "test_function",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "description": "Test function",
        },
        outputs={"result": OutputDefinition(type="str")},
    )

    plan = Plan(steps=[step], artifacts_dir="test_artifacts")

    with patch.object(executor, "_execute_step", side_effect=Exception("Test error")):
        with pytest.raises(ExecutorError):
            executor.execute_plan(plan)


def test_cleanup_and_artifacts(executor):
    """Test cleanup and artifacts handling."""
    step = PlanStep(
        name="test_step",
        tool="generate_code",
        inputs={
            "function_name": "test_function",
            "parameters": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "description": "Return the absolute value of the input number x. For example, if x is -5, return 5.",
        },
        outputs={"result": OutputDefinition(type="str")},
    )

    plan = Plan(steps=[step], artifacts_dir="test_artifacts")

    result = executor.execute_plan(plan)
    assert result is not None

    # Check artifacts directory exists
    assert os.path.exists("test_artifacts")
