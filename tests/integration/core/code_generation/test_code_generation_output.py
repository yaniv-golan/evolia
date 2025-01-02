"""Integration tests for code generation output handling."""
import pytest
from unittest.mock import patch
from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.models import Parameter
from evolia.utils.exceptions import (
    CodeGenerationError,
    APIRateLimitError,
    TemplateError,
)


@pytest.fixture
def code_generator():
    """Create a code generator for testing."""
    config = CodeGenerationConfig(
        model="gpt-4", api_key="test-key", temperature=0.2, max_tokens=1000
    )
    return CodeGenerator(config)


@pytest.fixture
def mock_responses():
    """Mock responses for testing."""
    return {
        "success": {
            "code": "def process_data(data: list) -> dict:\n    return {'count': len(data)}",
            "function_name": "process_data",
            "parameters": [{"name": "data", "type": "list"}],
            "return_type": "dict",
            "validation_results": {"syntax_valid": True, "security_issues": []},
        },
        "invalid_syntax": {
            "code": "def bad_function(x:\n    return x",
            "validation_results": {"syntax_valid": False, "security_issues": []},
        },
        "security_issue": {
            "code": "import os\ndef dangerous(x):\n    os.system(x)",
            "validation_results": {
                "syntax_valid": True,
                "security_issues": ["Contains system call"],
            },
        },
    }


def test_successful_generation(code_generator, mock_responses):
    """Test successful code generation output."""
    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=mock_responses["success"],
    ):
        response = code_generator.generate(
            prompt_template="Generate a function to {description}",
            template_vars={
                "description": "Process input data",
                "parameters": [
                    Parameter(name="data", type="list", description="Input data")
                ],
                "return_type": "dict",
            },
            schema={"type": "object", "properties": {}},
        )

        assert response is not None
        assert "code" in response
        assert response["function_name"] == "process_data"
        assert response["validation_results"]["syntax_valid"]
        assert not response["validation_results"]["security_issues"]


def test_invalid_syntax_output(code_generator, mock_responses):
    """Test handling of invalid syntax in output."""
    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=mock_responses["invalid_syntax"],
    ):
        with pytest.raises(CodeGenerationError) as exc_info:
            code_generator.generate(
                prompt_template="Generate a function with syntax error",
                template_vars={},
                schema={"type": "object", "properties": {}},
            )
        assert "syntax" in str(exc_info.value).lower()


def test_security_issue_output(code_generator, mock_responses):
    """Test handling of security issues in output."""
    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=mock_responses["security_issue"],
    ):
        with pytest.raises(CodeGenerationError) as exc_info:
            code_generator.generate(
                prompt_template="Generate a dangerous function",
                template_vars={},
                schema={"type": "object", "properties": {}},
            )
        assert "security" in str(exc_info.value).lower()


def test_edge_cases(code_generator):
    """Test handling of various edge cases in responses."""
    test_cases = [
        ({}, "empty response"),
        ({"validation_results": {}}, "missing code"),
        ({"code": "", "validation_results": {}}, "empty code"),
        ({"code": None, "validation_results": {}}, "null code"),
        (
            {"code": "def func():\n    pass", "validation_results": None},
            "missing validation",
        ),
    ]

    for response, expected_error in test_cases:
        with patch(
            "evolia.core.code_generator.call_openai_structured", return_value=response
        ):
            with pytest.raises(CodeGenerationError) as exc_info:
                code_generator.generate(
                    prompt_template="Generate a function",
                    template_vars={},
                    schema={"type": "object", "properties": {}},
                )
            assert expected_error in str(exc_info.value).lower()


def test_invalid_type_output(code_generator, mock_responses):
    """Test handling of invalid type in output."""
    invalid_response = mock_responses["success"].copy()
    invalid_response["parameters"][0]["type"] = "invalid_type"

    with patch(
        "evolia.core.code_generator.call_openai_structured",
        return_value=invalid_response,
    ):
        with pytest.raises(CodeGenerationError) as exc_info:
            code_generator.generate(
                prompt_template="Generate a function",
                template_vars={
                    "description": "Process data",
                    "parameters": [
                        Parameter(name="data", type="list", description="Input")
                    ],
                    "return_type": "dict",
                },
                schema={"type": "object", "properties": {}},
            )
        assert "type" in str(exc_info.value).lower()


def test_rate_limit_handling(code_generator):
    """Test handling of API rate limits."""

    def raise_rate_limit(*args, **kwargs):
        raise APIRateLimitError("Rate limit exceeded")

    with patch(
        "evolia.core.code_generator.call_openai_structured",
        side_effect=raise_rate_limit,
    ):
        with pytest.raises(APIRateLimitError) as exc_info:
            code_generator.generate(
                prompt_template="Generate a function",
                template_vars={},
                schema={"type": "object", "properties": {}},
            )
        assert "rate limit" in str(exc_info.value).lower()


def test_template_validation(code_generator):
    """Test validation of template variables."""
    test_cases = [
        # Missing required variable
        ("Generate a function to {description}", {}, "missing required variable"),
        # Invalid variable type
        ("Generate a {type} function", {"type": 123}, "variable type must be string"),
        # Unknown variable
        ("Generate a function", {"unknown": "value"}, "unknown template variable"),
    ]

    for template, vars, expected_error in test_cases:
        with pytest.raises(TemplateError) as exc_info:
            code_generator.generate(
                prompt_template=template,
                template_vars=vars,
                schema={"type": "object", "properties": {}},
            )
        assert expected_error in str(exc_info.value).lower()


def test_api_error_handling(code_generator):
    """Test handling of various API errors."""
    error_cases = [
        ("Invalid API key", "authentication"),
        ("Model overloaded", "server overload"),
        ("Bad gateway", "service unavailable"),
        ("Context length exceeded", "input too long"),
    ]

    for error_msg, expected_phrase in error_cases:

        def raise_api_error(*args, **kwargs):
            raise Exception(error_msg)

        with patch(
            "evolia.core.code_generator.call_openai_structured",
            side_effect=raise_api_error,
        ):
            with pytest.raises(CodeGenerationError) as exc_info:
                code_generator.generate(
                    prompt_template="Generate a function",
                    template_vars={},
                    schema={"type": "object", "properties": {}},
                )
            assert expected_phrase in str(exc_info.value).lower()
