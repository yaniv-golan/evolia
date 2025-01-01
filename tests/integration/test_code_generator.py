"""Integration tests for the code generator module."""

import os
import pytest
from src.evolia.core.code_generator import (
    CodeGenerator, CodeGenerationConfig, call_openai_structured
)
from src.evolia.core.function_generator import FunctionGenerator
from src.evolia.models.models import CodeGenerationRequest, Parameter

@pytest.fixture
def openai_api_key():
    """Ensure OPENAI_API_KEY is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key

@pytest.mark.integration
def test_simple_math_function(openai_api_key):
    """Test generating a simple math function."""
    config = CodeGenerationConfig(api_key=openai_api_key)
    code_generator = CodeGenerator(config)
    generator = FunctionGenerator(code_generator)
    
    response = generator.generate_function(
        requirements="Add two numbers and return the result",
        function_name="add_numbers",
        parameters=[
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ],
        return_type="int"
    )
    
    assert response["validation_results"]["syntax_valid"]
    assert "def add_numbers(a: int, b: int) -> int:" in response["code"]
    assert "return" in response["code"]

@pytest.mark.integration
def test_string_processing_function(openai_api_key):
    """Test generating a string processing function."""
    config = CodeGenerationConfig(api_key=openai_api_key)
    code_generator = CodeGenerator(config)
    generator = FunctionGenerator(code_generator)
    
    response = generator.generate_function(
        requirements="Convert a string to uppercase and append a suffix",
        function_name="process_string",
        parameters=[
            {"name": "text", "type": "str"},
            {"name": "suffix", "type": "str"}
        ],
        return_type="str"
    )
    
    assert response["validation_results"]["syntax_valid"]
    assert "def process_string(text: str, suffix: str) -> str:" in response["code"]

@pytest.mark.integration
def test_json_handling_function(openai_api_key):
    """Test generating a JSON processing function."""
    config = CodeGenerationConfig(
        api_key=openai_api_key,
        allowed_modules={"json"}
    )
    code_generator = CodeGenerator(config)
    generator = FunctionGenerator(code_generator)
    
    response = generator.generate_function(
        requirements="Parse a JSON string and extract a specific field",
        function_name="extract_field",
        parameters=[
            {"name": "json_str", "type": "str"},
            {"name": "field_name", "type": "str"}
        ],
        return_type="dict",
        context='''Example format:
def example(json_str: str, field_name: str) -> dict:
    try:
        data = json.loads(json_str)
        return {"result": data.get(field_name)}
    except Exception as e:
        return {"error": str(e)}'''
    )
    
    assert response["validation_results"]["syntax_valid"]
    assert "import json" in response["code"]
    assert "try:" in response["code"]
    assert "except" in response["code"]

@pytest.mark.integration
def test_complex_type_hints(openai_api_key):
    """Test generating a function with complex type hints."""
    config = CodeGenerationConfig(api_key=openai_api_key)
    code_generator = CodeGenerator(config)
    generator = FunctionGenerator(code_generator)
    
    response = generator.generate_function(
        requirements="Filter a list of dictionaries based on a key-value pair",
        function_name="filter_dicts",
        parameters=[
            {"name": "items", "type": "list"},
            {"name": "key", "type": "str"},
            {"name": "value", "type": "Any"}
        ],
        return_type="list"
    )
    
    assert response["validation_results"]["syntax_valid"]
    assert "def filter_dicts" in response["code"]

@pytest.mark.integration
def test_docstring_generation(openai_api_key):
    """Test that generated code includes proper docstrings."""
    config = CodeGenerationConfig(api_key=openai_api_key)
    code_generator = CodeGenerator(config)
    generator = FunctionGenerator(code_generator)
    
    response = generator.generate_function(
        requirements="Calculate the factorial of a number recursively",
        function_name="factorial",
        parameters=[{"name": "n", "type": "int"}],
        return_type="int"
    )
    
    assert response["validation_results"]["syntax_valid"]
    assert '"""' in response["code"]  # Should contain docstring
    assert "Parameters:" in response["code"] or "Args:" in response["code"]  # Accept either format
    assert "Returns:" in response["code"] 