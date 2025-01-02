"""Integration tests using live OpenAI API."""
import pytest
from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.core.function_generator import FunctionGenerator
from evolia.models.models import Parameter
from evolia.utils.exceptions import CodeGenerationError


@pytest.mark.integration
def test_live_code_generation(openai_api_key):
    """Test code generation using live API."""
    config = CodeGenerationConfig(
        api_key=openai_api_key,
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        max_tokens=1000,
        allowed_modules={"math", "typing"},
        allowed_builtins={"len", "str", "int", "float"},
    )
    code_generator = CodeGenerator(config)
    function_generator = FunctionGenerator(code_generator)

    # Test simple function generation
    response = function_generator.generate_function(
        requirements="Generate a function that calculates the square of a number.",
        function_name="calculate_square",
        parameters=[
            Parameter(name="number", type="float", description="Number to square")
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

    # Test functionality
    square_func = namespace["calculate_square"]
    assert square_func(2.0) == 4.0
    assert square_func(3.0) == 9.0
    assert square_func(-2.0) == 4.0  # Should handle negative numbers


@pytest.mark.integration
def test_live_code_generation_complex(openai_api_key):
    """Test generation of more complex functions using live API."""
    config = CodeGenerationConfig(
        api_key=openai_api_key,
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        max_tokens=1000,
        allowed_modules={"typing", "datetime", "re"},
        allowed_builtins={"len", "str", "int", "float", "list", "dict"},
    )
    code_generator = CodeGenerator(config)
    function_generator = FunctionGenerator(code_generator)

    # Test complex function generation
    response = function_generator.generate_function(
        requirements="""
        Generate a function that:
        1. Takes a list of datetime strings in various formats
        2. Converts them to ISO format (YYYY-MM-DD)
        3. Returns only unique dates in sorted order
        4. Handles invalid dates gracefully by skipping them
        """,
        function_name="normalize_dates",
        parameters=[
            Parameter(
                name="date_strings",
                type="List[str]",
                description="List of date strings in various formats",
            )
        ],
        return_type="List[str]",
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

    # Test functionality
    normalize_func = namespace["normalize_dates"]
    test_dates = [
        "2024/01/15",
        "01-15-2024",
        "2024/01/15",  # Duplicate
        "invalid_date",
        "2024/01/16",
    ]
    result = normalize_func(test_dates)

    assert isinstance(result, list)
    assert all(isinstance(date, str) for date in result)
    assert len(result) == 2  # Should have 2 unique valid dates
    assert result == ["2024-01-15", "2024-01-16"]  # Should be sorted
