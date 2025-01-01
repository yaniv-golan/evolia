"""Integration tests for output types and references handling."""

import pytest
import os
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.core.executor2 import Executor2
from evolia.core.prompts import BASE_SYSTEM_PROMPT, BASE_VALIDATION_SCHEMA
from evolia.models.models import Parameter, PlanStep

def setup_test_env():
    """Set up test environment with temporary directories."""
    temp_dir = tempfile.mkdtemp()
    artifacts_dir = Path(temp_dir) / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return temp_dir, artifacts_dir

def cleanup_test_env(temp_dir: str):
    """Clean up test environment."""
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
        "allowed_modules": {
            'typing', 'datetime', 'json', 'logging', 're', 'math',
            'collections', 'itertools', 'functools'
        },
        "allowed_builtins": {
            'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
            'min', 'max', 'sum', 'any', 'all', 'isinstance', 'hasattr'
        }
    }
    return Executor2(config)

def test_single_output_generation_and_execution(code_generator, executor):
    """Test generation and execution of code with a single output."""
    temp_dir, artifacts_dir = setup_test_env()
    try:
        # Define test parameters
        prompt_template = """Create a function that adds two numbers and returns the result as a string.
        The function should be named 'add_numbers' and take two integer parameters 'a' and 'b'.
        The output should be named 'result' with type 'str'."""
        
        template_vars = {
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
        
        # Generate code
        response = code_generator.generate(
            prompt_template=prompt_template,
            template_vars=template_vars,
            schema=BASE_VALIDATION_SCHEMA,
            system_prompt=BASE_SYSTEM_PROMPT
        )
        
        assert response["outputs"]["result"]["type"] == "str"
        assert response["outputs"]["result"]["reference"] == "$add_numbers.result"
        
        # Write code to file
        script_file = Path(artifacts_dir) / "add_numbers.py"
        script_file.write_text(response["code"])
        
        # Execute generated code
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
        
    finally:
        cleanup_test_env(temp_dir)

def test_multiple_outputs_generation_and_execution(code_generator, executor):
    """Test generation and execution of code with multiple outputs."""
    temp_dir, artifacts_dir = setup_test_env()
    try:
        # Define test parameters
        prompt_template = """Create a function that processes a dictionary of numbers.
        The function should be named 'process_numbers' and take a dictionary parameter 'data'.
        It should return:
        1. The sum of values as 'total'
        2. The count of values as 'count'
        Both outputs should be strings."""
        
        template_vars = {
            "function_name": "process_numbers",
            "parameters": [
                Parameter(name="data", type="dict", description="Dictionary of numbers")
            ],
            "outputs": {
                "total": {
                    "type": "str",
                    "reference": "$process_numbers.total"
                },
                "count": {
                    "type": "str",
                    "reference": "$process_numbers.count"
                }
            }
        }
        
        # Generate code
        response = code_generator.generate(
            prompt_template=prompt_template,
            template_vars=template_vars,
            schema=BASE_VALIDATION_SCHEMA,
            system_prompt=BASE_SYSTEM_PROMPT
        )
        
        assert "total" in response["outputs"]
        assert "count" in response["outputs"]
        assert response["outputs"]["total"]["type"] == "str"
        assert response["outputs"]["count"]["type"] == "str"
        
        # Write code to file
        script_file = Path(artifacts_dir) / "process_numbers.py"
        script_file.write_text(response["code"])
        
        # Execute generated code
        step = PlanStep(
            name="process_numbers",
            tool="python",
            inputs={
                "script_file": str(script_file.relative_to(temp_dir)),
                "data": {"a": 1, "b": 2, "c": 3}
            },
            outputs=response["outputs"]
        )
        result = executor.execute_code(step, str(artifacts_dir))
        
        assert isinstance(result["total"], str)
        assert isinstance(result["count"], str)
        assert result["total"] == "6"
        assert result["count"] == "3"
        
    finally:
        cleanup_test_env(temp_dir)

def test_output_reference_in_subsequent_step(code_generator, executor):
    """Test using output reference from one step in a subsequent step."""
    temp_dir, artifacts_dir = setup_test_env()
    try:
        # First step: Generate numbers
        step1_prompt = """Create a function that generates a range of numbers.
        The function should be named 'generate_range' and take 'start' and 'end' parameters.
        Return the range as a comma-separated string."""
        
        step1_vars = {
            "function_name": "generate_range",
            "parameters": [
                Parameter(name="start", type="int", description="Start number"),
                Parameter(name="end", type="int", description="End number")
            ],
            "outputs": {
                "numbers": {
                    "type": "str",
                    "reference": "$generate_range.numbers"
                }
            }
        }
        
        step1_response = code_generator.generate(
            prompt_template=step1_prompt,
            template_vars=step1_vars,
            schema=BASE_VALIDATION_SCHEMA,
            system_prompt=BASE_SYSTEM_PROMPT
        )
        
        # Write first step code to file
        script_file1 = Path(artifacts_dir) / "generate_range.py"
        script_file1.write_text(step1_response["code"])
        
        # Second step: Process the numbers
        step2_prompt = """Create a function that processes a comma-separated string of numbers.
        The function should be named 'process_string' and take a string parameter 'numbers'.
        Return the sum of the numbers as a string."""
        
        step2_vars = {
            "function_name": "process_string",
            "parameters": [
                Parameter(name="numbers", type="str", description="Comma-separated numbers")
            ],
            "outputs": {
                "sum": {
                    "type": "str",
                    "reference": "$process_string.sum"
                }
            }
        }
        
        step2_response = code_generator.generate(
            prompt_template=step2_prompt,
            template_vars=step2_vars,
            schema=BASE_VALIDATION_SCHEMA,
            system_prompt=BASE_SYSTEM_PROMPT
        )
        
        # Write second step code to file
        script_file2 = Path(artifacts_dir) / "process_string.py"
        script_file2.write_text(step2_response["code"])
        
        # Execute first step
        step1 = PlanStep(
            name="generate_range",
            tool="python",
            inputs={
                "script_file": str(script_file1.relative_to(temp_dir)),
                "start": 1,
                "end": 5
            },
            outputs=step1_response["outputs"]
        )
        step1_result = executor.execute_code(step1, str(artifacts_dir))
        
        assert isinstance(step1_result["numbers"], str)
        
        # Execute second step using first step's output
        step2 = PlanStep(
            name="process_string",
            tool="python",
            inputs={
                "script_file": str(script_file2.relative_to(temp_dir)),
                "numbers": step1_result["numbers"]
            },
            outputs=step2_response["outputs"]
        )
        step2_result = executor.execute_code(step2, str(artifacts_dir))
        
        assert isinstance(step2_result["sum"], str)
        assert step2_result["sum"] == "15"  # Sum of numbers 1 to 5
        
    finally:
        cleanup_test_env(temp_dir)

def test_invalid_output_type_handling(code_generator, executor):
    """Test handling of invalid output types during execution."""
    temp_dir, artifacts_dir = setup_test_env()
    try:
        # Define test with mismatched output type
        prompt_template = """Create a function that adds numbers but returns an integer instead of string.
        The function should be named 'add_wrong' and take two integer parameters."""
        
        template_vars = {
            "function_name": "add_wrong",
            "parameters": [
                Parameter(name="a", type="int", description="First number"),
                Parameter(name="b", type="int", description="Second number")
            ],
            "outputs": {
                "result": {
                    "type": "str",  # Expecting string
                    "reference": "$add_wrong.result"
                }
            }
        }
        
        response = code_generator.generate(
            prompt_template=prompt_template,
            template_vars=template_vars,
            schema=BASE_VALIDATION_SCHEMA,
            system_prompt=BASE_SYSTEM_PROMPT
        )
        
        # Write code to file
        script_file = Path(artifacts_dir) / "add_wrong.py"
        script_file.write_text(response["code"])
        
        # Execute should raise error due to type mismatch
        step = PlanStep(
            name="add_wrong",
            tool="python",
            inputs={
                "script_file": str(script_file.relative_to(temp_dir)),
                "a": 1,
                "b": 2
            },
            outputs=response["outputs"]
        )
        with pytest.raises(Exception) as exc_info:
            executor.execute_code(step, str(artifacts_dir))
        
        assert "type" in str(exc_info.value).lower()
        
    finally:
        cleanup_test_env(temp_dir) 