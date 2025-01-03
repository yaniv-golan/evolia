"""Integration tests for the code generator module."""

import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from evolia.core.code_generator import CodeGenerationConfig, CodeGenerator
from evolia.core.function_generator import FunctionGenerator
from evolia.models.models import Parameter
from evolia.utils.exceptions import CodeGenerationError


@pytest.fixture
def mock_responses():
    """Mock responses for testing."""
    return {
        "math_function": {
            "code": """def add_positive_numbers(a: float, b: float) -> float:
    if a < 0 or b < 0:
        raise ValueError("Numbers must be positive")
    return a + b""",
            "validation_results": {"syntax_valid": True, "security_issues": []},
        },
        "string_function": {
            "code": """def clean_and_title(text: str) -> str:
    return " ".join(text.split()).title()""",
            "validation_results": {"syntax_valid": True, "security_issues": []},
        },
        "complex_function": {
            "code": """from typing import List, Dict, Any, Callable
def filter_and_transform(items: List[Dict[str, Any]], key: str, value: Any, transform_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [transform_func(item) for item in items if key in item and item[key] == value]""",
            "validation_results": {"syntax_valid": True, "security_issues": []},
        },
    }


@pytest.fixture
def function_generator():
    """Create a function generator instance."""
    config = CodeGenerationConfig(
        api_key="test-key",
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        max_tokens=1000,
        allowed_modules={"math", "typing", "datetime", "re", "json"},
        allowed_builtins={"len", "str", "int", "float", "list", "dict", "set"},
    )
    code_generator = CodeGenerator(config)
    return FunctionGenerator(code_generator)


def test_math_function_generation(function_generator, mock_responses):
    """Test generating a math function."""
    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=mock_responses["math_function"],
    ):
        response = function_generator.generate_function(
            requirements="Generate a function that adds two positive numbers.",
            function_name="add_positive_numbers",
            parameters=[
                Parameter(name="a", type="float", description="First number"),
                Parameter(name="b", type="float", description="Second number"),
            ],
            return_type="float",
        )

        # Verify response
        assert response is not None
        assert "code" in response
        assert response["validation_results"]["syntax_valid"]
        assert not response["validation_results"]["security_issues"]

        # Execute the generated code
        code = response["code"]
        namespace = {}
        exec(code, namespace)

        # Verify functionality
        add_func = namespace["add_positive_numbers"]
        assert add_func(2.5, 3.5) == 6.0

        # Verify error handling
        with pytest.raises(ValueError) as exc_info:
            add_func(-1.0, 2.0)
        assert "must be positive" in str(exc_info.value).lower()


def test_string_processing_function(function_generator, mock_responses):
    """Test generating a string processing function."""
    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=mock_responses["string_function"],
    ):
        response = function_generator.generate_function(
            requirements="Convert a string to title case and remove extra whitespace. Handle empty strings gracefully.",
            function_name="clean_and_title",
            parameters=[
                Parameter(name="text", type="str", description="Input text to process")
            ],
            return_type="str",
        )

        assert response["validation_results"]["syntax_valid"]
        assert "def clean_and_title" in response["code"]

        # Execute and test the function
        namespace = {}
        exec(response["code"], namespace)
        clean_func = namespace["clean_and_title"]

        assert clean_func("hello  world") == "Hello World"
        assert clean_func("") == ""
        assert clean_func("  multiple   spaces  ") == "Multiple Spaces"


def test_complex_function_generation(function_generator, mock_responses):
    """Test generating a more complex function with type hints."""
    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=mock_responses["complex_function"],
    ):
        response = function_generator.generate_function(
            requirements="""Filter and transform a list of dictionaries:
            1. Keep only items where the specified key exists and matches the value
            2. Transform matching items using the transform_func
            3. Maintain order of filtered items""",
            function_name="filter_and_transform",
            parameters=[
                Parameter(
                    name="items",
                    type="List[Dict[str, Any]]",
                    description="List of dictionaries to filter",
                ),
                Parameter(name="key", type="str", description="Key to filter on"),
                Parameter(name="value", type="Any", description="Value to match"),
                Parameter(
                    name="transform_func",
                    type="Callable[[Dict[str, Any]], Dict[str, Any]]",
                    description="Function to transform matching items",
                ),
            ],
            return_type="List[Dict[str, Any]]",
        )

        assert response["validation_results"]["syntax_valid"]
        assert "from typing import" in response["code"]
        assert "List[Dict[str, Any]]" in response["code"]
        assert "Callable" in response["code"]

        # Execute and test the function
        namespace = {}
        exec(response["code"], namespace)
        filter_func = namespace["filter_and_transform"]

        # Test data
        test_data = [
            {"id": 1, "name": "test1"},
            {"id": 2, "value": "keep"},
            {"id": 3, "value": "keep"},
            {"id": 4, "name": "test4"},
        ]

        def transform(item: Dict[str, Any]) -> Dict[str, Any]:
            return {**item, "transformed": True}

        result = filter_func(test_data, "value", "keep", transform)
        assert len(result) == 2
        assert all(item["transformed"] for item in result)
        assert all(item["value"] == "keep" for item in result)
