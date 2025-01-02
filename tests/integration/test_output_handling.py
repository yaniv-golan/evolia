"""Integration tests for output types and references handling."""

import pytest
import os
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.core.executor2 import Executor2
from evolia.models.models import Parameter, PlanStep

@pytest.fixture
def test_env():
    """Set up and clean up test environment."""
    temp_dir = tempfile.mkdtemp()
    artifacts_dir = Path(temp_dir) / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    yield temp_dir, artifacts_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def code_generator():
    """Create a CodeGenerator instance for testing."""
    config = CodeGenerationConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-2024-08-06",
        temperature=0.2
    )
    return CodeGenerator(config)

@pytest.fixture
def executor():
    """Create an Executor2 instance for testing."""
    config = {
        "allowed_modules": {'typing', 'json', 'math'},
        "allowed_builtins": {'str', 'int', 'float', 'list', 'dict'}
    }
    return Executor2(config)

def test_basic_output_handling(code_generator, executor, test_env):
    """Test basic output generation and execution."""
    temp_dir, artifacts_dir = test_env
    
    # Generate a simple function
    response = code_generator.generate(
        prompt_template="Create a function that adds two numbers and returns the result as a string.",
        template_vars={
            "function_name": "add_numbers",
            "parameters": [
                Parameter(name="a", type="int", description="First number"),
                Parameter(name="b", type="int", description="Second number")
            ],
            "outputs": {
                "result": {
                    "type": "str",
                    "reference": "$add_numbers.result"
                }
            }
        }
    )
    
    # Verify output structure
    assert "result" in response["outputs"]
    assert response["outputs"]["result"]["type"] == "str"
    
    # Execute the function
    script_file = artifacts_dir / "add_numbers.py"
    script_file.write_text(response["code"])
    
    step = PlanStep(
        name="add_numbers",
        tool="python",
        inputs={
            "script_file": str(script_file.relative_to(temp_dir)),
            "a": 5,
            "b": 3
        },
        outputs=response["outputs"]
    )
    result = executor.execute_code(step, str(artifacts_dir))
    
    assert isinstance(result["result"], str)
    assert result["result"] == "8"

def test_output_references(code_generator, executor, test_env):
    """Test output references between steps."""
    temp_dir, artifacts_dir = test_env
    
    # First step: Generate a list
    step1_response = code_generator.generate(
        prompt_template="Create a function that returns a list of numbers as a string.",
        template_vars={
            "function_name": "get_numbers",
            "parameters": [],
            "outputs": {
                "numbers": {
                    "type": "str",
                    "reference": "$get_numbers.numbers"
                }
            }
        }
    )
    
    script1 = artifacts_dir / "get_numbers.py"
    script1.write_text(step1_response["code"])
    
    # Second step: Process the list
    step2_response = code_generator.generate(
        prompt_template="Create a function that sums numbers from a string.",
        template_vars={
            "function_name": "sum_numbers",
            "parameters": [
                Parameter(name="numbers", type="str", description="Numbers to sum")
            ],
            "outputs": {
                "sum": {
                    "type": "str",
                    "reference": "$sum_numbers.sum"
                }
            }
        }
    )
    
    script2 = artifacts_dir / "sum_numbers.py"
    script2.write_text(step2_response["code"])
    
    # Execute steps
    step1 = PlanStep(
        name="get_numbers",
        tool="python",
        inputs={"script_file": str(script1.relative_to(temp_dir))},
        outputs=step1_response["outputs"]
    )
    step1_result = executor.execute_code(step1, str(artifacts_dir))
    
    step2 = PlanStep(
        name="sum_numbers",
        tool="python",
        inputs={
            "script_file": str(script2.relative_to(temp_dir)),
            "numbers": step1_result["numbers"]
        },
        outputs=step2_response["outputs"]
    )
    step2_result = executor.execute_code(step2, str(artifacts_dir))
    
    assert isinstance(step2_result["sum"], str)
    assert int(step2_result["sum"]) > 0 