"""Unit tests for Executor2."""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from evolia.core.executor2 import Executor2, ExecutorError
from evolia.models.models import Plan, PlanStep, CodeGenerationRequest
from evolia.utils.exceptions import CodeGenerationError, SecurityViolationError

@pytest.fixture
def mock_config():
    """Basic config for testing."""
    return {
        "file_access": {
            "paths": {
                "ephemeral_base": "test_artifacts",
                "tools_base": "test_tools",
                "data_base": "test_data"
            }
        },
        "allowed_modules": {"json", "typing", "math"},
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "model": "gpt-4o-2024-08-06"
        }
    }

@pytest.fixture
def executor(mock_config, tmp_path):
    """Create Executor2 instance with test config."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return Executor2(mock_config, keep_artifacts=True, ephemeral_dir=str(tmp_path))

def test_executor_init(executor, mock_config, tmp_path):
    """Test Executor2 initialization."""
    assert executor.config == mock_config
    assert executor.keep_artifacts is True
    assert executor.ephemeral_dir == str(tmp_path)
    assert executor.allowed_modules == {"json", "typing", "math"}
    assert isinstance(executor.data_store, dict)
    assert executor.artifacts_dir == Path(tmp_path)

def test_executor_init_no_artifacts(mock_config):
    """Test Executor2 initialization without artifacts."""
    executor = Executor2(mock_config, keep_artifacts=False)
    assert executor.ephemeral_dir is None
    assert executor.artifacts_dir is None

def test_executor_init_missing_api_key(mock_config):
    """Test Executor2 initialization with missing API key."""
    with patch.dict(os.environ, clear=True):
        with pytest.raises(ValueError, match="Missing OpenAI API key in environment variable OPENAI_API_KEY"):
            Executor2(mock_config)

@pytest.fixture
def mock_code_generator():
    """Mock CodeGenerator for testing."""
    mock = Mock()
    mock.generate.return_value = {
        "code": "def test(): pass",
        "validation_results": {
            "syntax_valid": True,
            "name_matches": True,
            "params_match": True,
            "return_type_matches": True,
            "security_issues": []
        },
        "function_name": "test",
        "parameters": [],
        "return_type": "dict",
        "description": "Test function",
        "examples": [],
        "constraints": []
    }
    return mock

@pytest.fixture
def mock_code_fixer():
    """Mock CodeFixer for testing."""
    mock = Mock()
    mock.fix_code.return_value = {
        "code": "def test(): pass",
        "validation_results": {
            "syntax_valid": True,
            "security_issues": []
        }
    }
    return mock

def test_generate_code_success(executor, mock_code_generator):
    """Test successful code generation."""
    mock_code = "def test(): pass"
    mock_response = {
        "code": mock_code,
        "validation_results": {
            "syntax_valid": True,
            "name_matches": True,
            "params_match": True,
            "return_type_matches": True,
            "security_issues": []
        },
        "function_name": "test",
        "parameters": [],
        "return_type": "dict",
        "description": "Test function",
        "examples": [],
        "constraints": []
    }
    executor.code_generator.generate = Mock(return_value=mock_response)
    executor.function_generator.code_generator.generate = Mock(return_value=mock_response)
    with patch('evolia.core.executor2.validate_code_security', return_value=[]):
        request = CodeGenerationRequest(
            description="Test function",
            function_name="test",
            return_type="dict"
        )
        response = executor._generate_code(request)
        assert response.code == mock_code
        assert response.validation_results.syntax_valid is True
        assert response.function_name == "test"
        assert response.return_type == "dict"

def test_generate_code_validation_error(executor, mock_code_generator):
    """Test code generation with validation error."""
    mock_code_generator.generate.return_value = {
        "code": "def test_func(): pass",
        "validation_results": {
            "syntax_valid": False,
            "name_matches": False,
            "params_match": False,
            "return_type_matches": False,
            "security_issues": ["Unsafe code detected"]
        },
        "function_name": "test_func",
        "parameters": [],
        "return_type": "dict",
        "description": "Test function",
        "examples": [],
        "constraints": []
    }
    with patch('evolia.core.executor2.CodeGenerator', return_value=mock_code_generator):
        with patch('evolia.core.executor2.validate_code_security', side_effect=SecurityViolationError("Unsafe code")):
            request = CodeGenerationRequest(
                description="Test function",
                function_name="test_func",
                return_type="dict"
            )
            with pytest.raises(CodeGenerationError):
                executor._generate_code(request)

def test_execute_plan_step_generate_code(executor, mock_code_generator):
    """Test executing a generate_code plan step."""
    step = PlanStep(
        name="test_step",
        tool="generate_code",
        inputs={
            "function_name": "test",
            "description": "Test function",
            "return_type": "dict"
        }
    )
    with patch('evolia.core.executor2.CodeGenerator', return_value=mock_code_generator):
        result = executor._execute_step(step)
        assert isinstance(result, dict)
        assert 'code' in result
        assert isinstance(result['code'], str)
        assert 'function_name' in result
        assert result['function_name'] == 'test'

def test_execute_plan_step_system_tool(executor):
    """Test executing a system tool plan step."""
    step = PlanStep(
        name="test_step",
        tool="test_tool",
        inputs={"input": "test"}
    )
    mock_tool = Mock()
    mock_tool.execute.return_value = {"output": "success"}
    with patch.object(executor, '_load_system_tool', return_value=mock_tool):
        result = executor._execute_step(step)
        assert result == {"output": "success"}

def test_execute_plan_step_invalid_tool(executor):
    """Test executing a plan step with invalid tool."""
    step = PlanStep(
        name="test_step",
        tool="invalid_tool",
        inputs={}
    )
    with pytest.raises(ExecutorError, match="Failed to execute step test_step: Invalid tool type: invalid_tool"):
        executor._execute_step(step)

def test_execute_plan(executor, mock_code_generator):
    """Test executing a complete plan."""
    mock_code = "def test(): pass"
    mock_response = {
        "code": mock_code,
        "validation_results": {
            "syntax_valid": True,
            "name_matches": True,
            "params_match": True,
            "return_type_matches": True,
            "security_issues": []
        },
        "function_name": "test",
        "parameters": [],
        "return_type": "dict",
        "description": "Test function",
        "examples": [],
        "constraints": []
    }
    executor.code_generator.generate = Mock(return_value=mock_response)
    executor.function_generator.code_generator.generate = Mock(return_value=mock_response)
    plan = Plan(steps=[
        PlanStep(
            name="step1",
            tool="generate_code",
            inputs={
                "function_name": "test",
                "description": "Test function",
                "return_type": "dict"
            },
            outputs={}
        )
    ])
    with patch('evolia.core.executor2.validate_code_security', return_value=[]):
        results = executor.execute_plan(plan)
        assert isinstance(results, dict)
        assert "step1" in results
        step_result = results["step1"]
        assert isinstance(step_result, dict)
        assert step_result["code"] == mock_code
        assert step_result["function_name"] == "test"

def test_cleanup(executor, tmp_path):
    """Test cleanup of artifacts."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    executor.generated_files.append(str(test_file))
    executor.keep_artifacts = False  # Ensure cleanup is performed
    executor.cleanup()
    assert not test_file.exists()

def test_no_cleanup_when_keep_artifacts(executor, tmp_path):
    """Test no cleanup when keep_artifacts is True."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    executor.generated_files.append(str(test_file))
    executor.keep_artifacts = True
    executor.cleanup()
    assert test_file.exists() 