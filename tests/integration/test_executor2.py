"""Integration tests for Executor2."""

import os
import pytest
from pathlib import Path
from typing import Dict, Any

from evolia.core.executor2 import Executor2
from evolia.models.models import Plan, PlanStep, Parameter

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
            "api_key_env_var": "OPENAI_API_KEY",
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.2  # Low temperature for consistent results
        }
    }

@pytest.fixture
def executor(config, tmp_path):
    """Create executor instance for testing."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return Executor2(config, ephemeral_dir=str(tmp_path))

@pytest.mark.integration
def test_simple_math_plan(executor):
    """Test generating and executing a simple math function."""
    plan = Plan(steps=[
        PlanStep(
            name="generate_math",
            tool="generate_code",
            inputs={
                "function_name": "add_numbers",
                "description": "Add two numbers and return the result",
                "parameters": [
                    Parameter(name="a", type="float", description="First number"),
                    Parameter(name="b", type="float", description="Second number")
                ],
                "return_type": "float"
            },
            outputs={"code_file": "math_func.py"}
        ),
        PlanStep(
            name="execute_math",
            tool="execute_code",
            inputs={
                "script_file": "${generate_math.code_file}",
                "a": 10.5,
                "b": 20.7
            },
            outputs={"result": "output"}
        )
    ])
    
    results = executor.execute_plan(plan)
    assert "generate_math" in results
    assert "execute_math" in results
    assert abs(results["execute_math"]["result"] - 31.2) < 0.1  # Allow small floating-point difference

@pytest.mark.integration
def test_string_processing_plan(executor):
    """Test generating and executing a string processing function."""
    plan = Plan(steps=[
        PlanStep(
            name="generate_string_processor",
            tool="generate_code",
            inputs={
                "function_name": "clean_text",
                "description": "Convert text to uppercase and remove all whitespace",
                "parameters": [
                    Parameter(name="text", type="str", description="Input text to process")
                ],
                "return_type": "str"
            },
            outputs={"code_file": "string_func.py"}
        ),
        PlanStep(
            name="execute_processor",
            tool="execute_code",
            inputs={
                "script_file": "${generate_string_processor.code_file}",
                "text": "Hello  World"
            },
            outputs={"result": "output"}
        )
    ])
    
    results = executor.execute_plan(plan)
    assert results["execute_processor"]["result"] == "HELLOWORLD"

@pytest.mark.integration
def test_multi_step_data_processing(executor):
    """Test a multi-step plan with data dependencies."""
    plan = Plan(steps=[
        # Step 1: Generate a function to create a list of numbers
        PlanStep(
            name="generate_list",
            tool="generate_code",
            inputs={
                "function_name": "create_number_list",
                "description": "Create a list of numbers from start to end",
                "parameters": [
                    Parameter(name="start", type="int", description="Start number"),
                    Parameter(name="end", type="int", description="End number")
                ],
                "return_type": "List[int]"
            },
            outputs={"code_file": "list_func.py"}
        ),
        # Step 2: Generate a function to calculate average
        PlanStep(
            name="generate_avg",
            tool="generate_code",
            inputs={
                "function_name": "calculate_average",
                "description": "Calculate the average of a list of numbers",
                "parameters": [
                    Parameter(name="numbers", type="List[int]", description="List of numbers")
                ],
                "return_type": "float"
            },
            outputs={"code_file": "avg_func.py"}
        ),
        # Step 3: Execute list creation
        PlanStep(
            name="create_list",
            tool="execute_code",
            inputs={
                "script_file": "${generate_list.code_file}",
                "start": 1,
                "end": 5
            },
            outputs={"result": "numbers"}
        ),
        # Step 4: Calculate average
        PlanStep(
            name="calc_average",
            tool="execute_code",
            inputs={
                "script_file": "${generate_avg.code_file}",
                "numbers": "${create_list.numbers}"
            },
            outputs={"result": "final_result"}
        )
    ])
    
    results = executor.execute_plan(plan)
    assert len(results) == 4
    assert isinstance(results["create_list"]["numbers"], list)
    assert abs(results["calc_average"]["final_result"] - 3.0) < 0.1 