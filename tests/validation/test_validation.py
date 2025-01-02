"""Tests for plan validation."""
import pytest
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock

from evolia.core.evolia import (
    generate_plan,
    validate_plan,
    PlanValidationError
)
from evolia.models import Plan, PlanStep, SystemTool

@pytest.fixture
def mock_config():
    """Test configuration."""
    return {
        'openai': {
            'model': 'gpt-4o-2024-08-06',
            'api_key_env_var': 'OPENAI_API_KEY',
            'temperature': 0.2
        },
        'validation': {
            'max_lines': 100,
            'max_complexity': 10,
            'banned_functions': ['eval', 'exec', 'compile']
        }
    }

@pytest.fixture
def mock_args():
    """Mock command line arguments."""
    args = argparse.Namespace()
    args.allow_read = []
    args.allow_write = []
    args.allow_create = []
    args.default_policy = "deny"
    args.keep_artifacts = True
    args.ephemeral_dir = str(Path.cwd() / "artifacts")
    return args

def test_generate_plan_validates_interfaces(mock_config, mock_args):
    """Test that generate_plan validates interfaces."""
    task = "Test task"
    system_tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "interface": {
                "function_name": "test_function",
                "parameters": [
                    {
                        "name": "x",
                        "type": "int",
                        "description": "First number"
                    }
                ],
                "return_type": "int",
                "description": "A test function"
            }
        }
    ]
    
    with patch('evolia.core.evolia.call_openai_structured') as mock_openai:
        mock_openai.return_value = {
            'steps': [
                {
                    'name': 'Test step',
                    'tool': 'test_tool',
                    'inputs': {'x': 5},
                    'outputs': {'result': {'type': 'int'}},
                    'allowed_read_paths': [],
                    'allowed_write_paths': [],
                    'allowed_create_paths': [],
                    'default_policy': 'deny'
                }
            ]
        }
        
        plan = generate_plan(task, system_tools, mock_config, mock_args)
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.interface_validation.matches_interface
        assert not step.interface_validation.validation_errors 