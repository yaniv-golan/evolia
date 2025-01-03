"""Tests for the Evolia core functionality"""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolia.core.evolia import (
    PlanExecutionError,
    PlanGenerationError,
    PlanValidationError,
    execute_plan,
    generate_plan,
    load_config,
    load_system_tools,
    validate_plan,
)
from evolia.core.executor2 import Executor2
from evolia.integrations.openai_structured import call_openai_structured
from evolia.models import OutputDefinition, Parameter, Plan, PlanStep, SystemTool
from evolia.utils.exceptions import CodeExecutionError


@pytest.fixture
def mock_args():
    """Create mock command line arguments."""
    return {
        "allow_read": [],
        "allow_write": [],
        "allow_create": [],
        "default_policy": "deny",
        "keep_artifacts": True,
        "ephemeral_dir": "artifacts",
    }


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        "openai": {
            "model": "gpt-4o-2024-08-06",
            "api_key": "test-key",
            "temperature": 0.7,
            "api_key_env_var": "OPENAI_API_KEY",
            "max_retries": 3,
            "retry_delay": 1,
        },
        "allowed_modules": ["math", "random", "typing"],
        "validation": {
            "max_lines": 100,
            "max_complexity": 10,
            "banned_functions": ["eval", "exec", "compile"],
            "max_syntax_lint_retries": 3,
            "max_runtime_retries": 3,
        },
        "file_access": {
            "paths": {
                "ephemeral_base": "test_artifacts",
                "tools_base": "test_tools",
                "data_base": "test_data",
            }
        },
    }


@pytest.fixture
def mock_system_tools():
    """Mock system tools for testing"""
    return {
        "test_function": {
            "name": "test_function",
            "description": "A test function",
            "interface": {
                "function_name": "test_function",
                "parameters": [
                    {"name": "x", "type": "int", "description": "First number"},
                    {"name": "y", "type": "int", "description": "Second number"},
                ],
                "return_type": "int",
                "description": "A test function that takes two numbers",
            },
            "metadata": {
                "loaded_at": "2024-12-30T20:35:18",
                "validated": True,
                "validation_errors": [],
                "last_execution": None,
                "execution_count": 0,
                "average_duration": 0.0,
                "success_rate": 0.0,
                "interface_version": "1.0",
                "permissions": {
                    "default_policy": "deny",
                    "allow_read": [],
                    "allow_write": [],
                    "allow_create": [],
                },
            },
        }
    }


def test_generate_plan(mock_config, mock_args, is_github_actions):
    """Test plan generation."""
    if is_github_actions:
        pytest.skip("Skipping OpenAI-dependent test in GitHub Actions")

    # Create test system tools
    test_tool = SystemTool(
        name="test_tool",
        description="A test tool",
        parameters=[Parameter(name="input", type="str", description="Input value")],
        outputs={"result": OutputDefinition(type="dict")},
        permissions={"read": [], "write": [], "create": []},
    )
    # Create dictionary with tool name as key and tool dict as value
    system_tools = {test_tool.name: test_tool}

    task = {
        "steps": [
            {
                "name": "Test step",
                "tool": "test_tool",
                "inputs": {"input": "test"},
                "outputs": {"result": OutputDefinition(type="dict")},
            }
        ]
    }

    with patch(
        "evolia.integrations.openai_structured.call_openai_structured"
    ) as mock_openai:
        mock_openai.return_value = task
        plan = generate_plan(task, system_tools, mock_config, mock_args)
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "test_tool"


def test_generate_plan_error(mock_config, mock_args, is_github_actions):
    """Test error handling in plan generation."""
    if is_github_actions:
        pytest.skip("Skipping OpenAI-dependent test in GitHub Actions")

    with patch(
        "evolia.core.evolia.call_openai_structured",
        side_effect=RuntimeError("API error"),
    ):
        with pytest.raises(PlanGenerationError) as exc_info:
            test_tool = SystemTool(
                name="test_tool",
                description="A test tool",
                parameters=[],
                outputs={},
                permissions={"read": [], "write": [], "create": []},
            )
            generate_plan(
                task="Test goal",
                system_tools={test_tool.name: test_tool},
                config=mock_config,
                args=mock_args,
            )

        assert "API error" in str(exc_info.value)


def test_load_config():
    """Test configuration loading"""
    mock_config = {
        "openai": {
            "model": "gpt-4o-2024-08-06",
            "api_key_env_var": "OPENAI_API_KEY",
            "max_retries": 3,
            "retry_delay": 1,
        },
        "allowed_modules": ["math", "random", "typing"],
        "validation": {
            "max_lines": 100,
            "max_complexity": 10,
            "banned_functions": ["eval", "exec", "compile"],
            "max_syntax_lint_retries": 3,
            "max_runtime_retries": 3,
        },
        "file_access": {
            "paths": {
                "ephemeral_base": "test_artifacts",
                "tools_base": "test_tools",
                "data_base": "test_data",
            },
            "permissions": {
                "default_policy": "deny",
                "allow_read": ["test_data"],
                "allow_write": ["test_artifacts"],
                "allow_create": ["test_artifacts"],
            },
        },
    }

    with patch("yaml.safe_load", return_value=mock_config):
        config = load_config()

        # Verify config structure
        assert isinstance(config, dict)
        assert "openai" in config
        assert config["openai"]["model"] == "gpt-4o-2024-08-06"
        assert config["openai"]["api_key_env_var"] == "OPENAI_API_KEY"
        assert config["openai"]["max_retries"] == 3

        assert "allowed_modules" in config
        assert "math" in config["allowed_modules"]
        assert "random" in config["allowed_modules"]

        assert "validation" in config
        assert config["validation"]["max_lines"] == 100
        assert config["validation"]["max_complexity"] == 10
        assert "eval" in config["validation"]["banned_functions"]

        assert "file_access" in config
        assert "paths" in config["file_access"]
        assert config["file_access"]["paths"]["ephemeral_base"] == "test_artifacts"
        assert "permissions" in config["file_access"]
        assert config["file_access"]["permissions"]["default_policy"] == "deny"


def test_load_system_tools():
    """Test loading system tools from JSON."""
    mock_json_data = [
        {
            "name": "example_tool",
            "description": "An example tool",
            "interface": {
                "function_name": "example_function",
                "parameters": [
                    {"name": "input", "type": "str", "description": "Input value"}
                ],
                "return_type": "dict",
                "description": "An example function that processes input",
            },
            "metadata": {
                "loaded_at": "2024-12-30T20:35:18",
                "validated": True,
                "validation_errors": [],
                "last_execution": None,
                "execution_count": 0,
                "average_duration": 0.0,
                "success_rate": 0.0,
                "interface_version": "1.0",
                "permissions": {
                    "default_policy": "deny",
                    "allow_read": [],
                    "allow_write": [],
                    "allow_create": [],
                },
            },
        }
    ]

    with patch("json.load", return_value=mock_json_data), patch(
        "pathlib.Path.exists", return_value=True
    ):
        tools = load_system_tools()
        assert len(tools) > 0

        for tool in tools.values():  # Iterate over dictionary values
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert isinstance(tool.name, str)
            assert isinstance(tool.description, str)
