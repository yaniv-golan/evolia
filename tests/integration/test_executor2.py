"""Integration tests for Executor2."""

import os
import pytest
from pathlib import Path
from typing import Dict, Any

from evolia.core.executor2 import Executor2
from evolia.models.models import Plan, PlanStep, CodeGenerationRequest
from evolia.utils.exceptions import CodeGenerationError

@pytest.fixture
def config():
    """Real config for integration testing."""
    return {
        "file_access": {
            "paths": {
                "ephemeral_base": "test_artifacts",
                "tools_base": "tools/system",
                "data_base": "test_data"
            }
        },
        "allowed_modules": {
            "json", "typing", "math", "datetime",
            "pathlib", "os.path", "re", "collections"
        },
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY"
        }
    }

@pytest.fixture
def executor(config, tmp_path):
    """Create real Executor2 instance."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return Executor2(config, keep_artifacts=True, ephemeral_dir=str(tmp_path))

def test_generate_simple_math_function(executor):
    """Test generating a simple math function."""
    request = CodeGenerationRequest(
        description="Create a function that adds two numbers",
        function_name="add_numbers",
        parameters=[
            {"name": "a", "type": "int", "description": "First number"},
            {"name": "b", "type": "int", "description": "Second number"}
        ],
        return_type="dict",
        constraints=["pure_function"]
    )
    
    response = executor._generate_code(request)
    assert response.code is not None
    assert response.validation_results.syntax_valid
    assert "def add_numbers" in response.code
    assert not response.validation_results.security_issues

def test_generate_string_processing_function(executor):
    """Test generating a string processing function."""
    request = CodeGenerationRequest(
        description="Create a function that counts word frequency in a text",
        function_name="word_frequency",
        parameters=[
            {"name": "text", "type": "str", "description": "Input text to analyze"}
        ],
        return_type="dict",
        constraints=["pure_function"]
    )
    
    response = executor._generate_code(request)
    assert response.code is not None
    assert response.validation_results.syntax_valid
    assert "def word_frequency" in response.code
    assert not response.validation_results.security_issues

def test_execute_multi_step_plan(executor):
    """Test executing a plan with multiple steps."""
    plan = Plan(steps=[
        PlanStep(
            name="generate_processor",
            tool="generate_code",
            inputs={
                "function_name": "process_data",
                "description": "Process a list of numbers by doubling each one",
                "parameters": [
                    {"name": "numbers", "type": "List[int]", "description": "List of numbers to process"}
                ],
                "return_type": "dict",
                "constraints": ["pure_function"]
            }
        ),
        PlanStep(
            name="execute_processor",
            tool="execute_code",
            inputs={
                "code": "$generate_processor.result.code",
                "test_cases": [
                    {
                        "input": [[1, 2, 3]],
                        "expected": {"result": [2, 4, 6]}
                    }
                ]
            }
        )
    ])
    
    results = executor.execute_plan(plan)
    assert len(results) == 2
    assert "generate_processor" in results
    assert "execute_processor" in results
    assert results["generate_processor"]["code"] is not None
    assert results["execute_processor"]["result"] == {"result": [2, 4, 6]}

def test_generate_and_fix_code(executor):
    """Test generating code that needs fixing."""
    # First generate code with a potential issue
    request = CodeGenerationRequest(
        description="Create a function that divides two numbers",
        function_name="divide_numbers",
        parameters=[
            {"name": "a", "type": "float", "description": "Numerator"},
            {"name": "b", "type": "float", "description": "Denominator"}
        ],
        return_type="dict",
        constraints=["handle_exceptions"]
    )
    
    response = executor._generate_code(request)
    assert response.code is not None
    assert "def divide_numbers" in response.code
    
    # Now execute it with a test case that should trigger division by zero
    step = PlanStep(
        name="test_division",
        tool="execute_code",
        inputs={
            "code": response.result.code,
            "test_cases": [
                {
                    "input": [1.0, 0.0],
                    "expected": {"error": "Division by zero"}
                }
            ]
        }
    )
    
    result = executor._execute_step(step)
    assert "error" in result["result"]

def test_generate_with_type_validation(executor):
    """Test generating code with complex type validation."""
    request = CodeGenerationRequest(
        description="Create a function that validates a user dictionary",
        function_name="validate_user",
        parameters=[
            {
                "name": "user",
                "type": "Dict[str, Any]",
                "description": "User data dictionary with name and age"
            }
        ],
        return_type="dict",
        constraints=["handle_exceptions"]
    )
    
    response = executor._generate_code(request)
    assert response.code is not None
    assert response.validation_results.syntax_valid
    assert "def validate_user" in response.code
    
    # Test the generated code with invalid input
    step = PlanStep(
        name="test_validation",
        tool="execute_code",
        inputs={
            "code": response.result.code,
            "test_cases": [
                {
                    "input": [{"name": 123, "age": "invalid"}],  # Invalid types
                    "expected": {"error": "Invalid user data"}
                }
            ]
        }
    )
    
    result = executor._execute_step(step)
    assert "error" in result["result"] 