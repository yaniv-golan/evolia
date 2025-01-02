"""Integration tests for Evolia."""
import pytest
from pathlib import Path
from unittest.mock import patch

from evolia.core.executor2 import Executor2
from evolia.models import Plan, PlanStep, Parameter, OutputDefinition

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "openai": {
            "model": "gpt-4o-2024-08-06",
            "api_key": "test-key",
            "temperature": 0.7
        }
    }

@pytest.fixture
def mock_responses():
    """Mock responses for testing."""
    return {
        "code": "def test_function(x: int) -> int:\n    \"\"\"Test function\"\"\"\n    return x * 2",
        "function_name": "test_function",
        "parameters": [
            {"name": "x", "type": "int"}
        ],
        "return_type": "int",
        "validation_results": {
            "syntax_valid": True,
            "security_issues": []
        },
        "outputs": {
            "result": {
                "type": "int",
                "reference": "$test_function.result"
            }
        }
    }

def test_basic_execution(config, mock_responses):
    """Test basic execution of a plan."""
    with patch('evolia.core.code_generator.call_openai_structured', return_value=mock_responses):
        executor = Executor2(
            config=config
        )
        
        plan = Plan(
            steps=[
                PlanStep(
                    name="test_step",
                    tool="generate_code",
                    inputs={
                        "function_name": "test_function",
                        "parameters": [{"name": "x", "type": "int"}],
                        "return_type": "int",
                        "description": "Test function"
                    },
                    outputs={"result": OutputDefinition(type="str")},
                    allowed_read_paths=[],
                    allowed_write_paths=["test_artifacts/tmp/test_function.py"],
                    allowed_create_paths=["test_artifacts/tmp/test_function.py"],
                    default_policy="deny"
                )
            ],
            artifacts_dir="test_artifacts"
        )
        
        result = executor.execute_plan(plan)
        assert result is not None
        assert len(result) > 0