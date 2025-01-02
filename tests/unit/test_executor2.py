"""Unit tests for executor2 module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from evolia.core.executor2 import Executor2
from evolia.models.models import PlanStep, Plan

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "openai": {
            "api_key": "test-key",
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "allowed_modules": ["os", "pathlib", "json"],
        "allowed_builtins": ["len", "str", "int", "float"],
        "file_access": {
            "ephemeral_dir": None,
            "tools_dir": None,
            "data_dir": None,
            "artifacts_dir": None
        }
    }

@pytest.fixture
def executor(mock_config, tmp_path):
    """Create an executor instance for testing."""
    mock_config["file_access"]["ephemeral_dir"] = str(tmp_path)
    mock_config["file_access"]["artifacts_dir"] = str(tmp_path)
    return Executor2(mock_config)

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="def test(x): return x + 1"))]
    )
    return mock_client

def test_code_generation_step(executor, mock_openai_client):
    """Test generating code for a step."""
    with patch("openai.OpenAI", return_value=mock_openai_client):
        step = PlanStep(
            tool="code_generator",
            inputs={"prompt": "Generate a function that adds 1 to a number"},
            outputs={"code": "test.py"}
        )
        result = executor.execute_step(step)
        assert result["code"] == "def test(x): return x + 1"

def test_execute_plan(executor, mock_openai_client):
    """Test executing a complete plan."""
    with patch("openai.OpenAI", return_value=mock_openai_client):
        plan = Plan(
            steps=[
                PlanStep(
                    tool="code_generator",
                    inputs={"prompt": "Generate a function that adds 1 to a number"},
                    outputs={"code": "test.py"}
                )
            ],
            artifacts_dir=str(Path(executor.artifacts_dir))
        )
        result = executor.execute_plan(plan)
        assert result["test.py"] == "def test(x): return x + 1"

def test_error_handling(executor, mock_openai_client):
    """Test error handling during execution."""
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API error")
    
    with patch("openai.OpenAI", return_value=mock_client):
        step = PlanStep(
            tool="code_generator",
            inputs={"prompt": "Generate a function"},
            outputs={"code": "test.py"}
        )
        with pytest.raises(Exception, match="API error"):
            executor.execute_step(step)

def test_cleanup_and_artifacts(executor, mock_openai_client):
    """Test cleanup behavior and keeping artifacts."""
    with patch("openai.OpenAI", return_value=mock_openai_client):
        test_file = Path(executor.ephemeral_dir) / "test.py"
        test_file.write_text("def test(): pass")
        
        # Test cleanup
        executor.cleanup()
        assert not test_file.exists()
        
        # Test keeping artifacts
        test_file.write_text("def test(): pass")
        executor.keep_artifacts = True
        executor.cleanup()
        assert test_file.exists() 