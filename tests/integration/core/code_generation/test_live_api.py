"""Integration tests using live OpenAI API."""
import logging

import pytest

from evolia.core.code_generator import CodeGenerationConfig, CodeGenerator
from evolia.core.function_generator import FunctionGenerator
from evolia.models.models import Parameter
from evolia.utils.exceptions import CodeGenerationError

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_live_code_generation(openai_api_key, is_github_actions):
    """Test code generation using live API."""
    if is_github_actions:
        pytest.skip("Skipping live API test in GitHub Actions")

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


@pytest.mark.integration
def test_live_code_generation_complex(openai_api_key, is_github_actions):
    """Test generation of more complex functions using live API."""
    if is_github_actions:
        pytest.skip("Skipping live API test in GitHub Actions")

    config = CodeGenerationConfig(
        api_key=openai_api_key,
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        max_tokens=1000,
        allowed_modules={"typing", "datetime", "re"},
        allowed_builtins={
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "sorted",
            "any",
            "all",
            "isinstance",
        },
    )
    code_generator = CodeGenerator(config)
    function_generator = FunctionGenerator(code_generator)

    # Test complex function generation
    response = function_generator.generate_function(
        requirements="""
        Generate a function that:
        1. Takes a list of datetime strings in various formats (e.g. "YYYY/MM/DD", "MM-DD-YYYY")
        2. Converts them to ISO format (YYYY-MM-DD)
        3. Returns only unique dates in sorted order
        4. Handles invalid dates gracefully by skipping them
        5. Must handle at least these formats:
           - "YYYY/MM/DD" (e.g. "2024/01/15")
           - "MM-DD-YYYY" (e.g. "01-15-2024")
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

    # Log the generated code and response
    logger.info("Generated code:\n%s", response["code"])
    logger.info("Full response:\n%s", response)

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
