"""Tests for function generator."""
import pytest
from unittest.mock import patch, MagicMock

from evolia.core.function_generator import FunctionGenerator
from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.models import Parameter


@pytest.fixture
def function_generator():
    """Create a function generator for testing."""
    code_generator = MagicMock(spec=CodeGenerator)
    return FunctionGenerator(code_generator)


def test_generate_function_template_variables():
    """Test generating function template variables."""
    mock_code_generator = MagicMock()
    mock_code_generator.config = MagicMock()
    mock_code_generator.config.allowed_modules = ["json", "os"]
    mock_code_generator.generate.return_value = {
        "function_name": "test_function",
        "parameters": ["x: int"],
        "return_type": "int",
        "requirements": "Create a function that adds two numbers",
        "context": "This is a test function",
        "validation_results": {
            "syntax_valid": True,
            "security_issues": [],
            "name_matches": True,
            "type_matches": True,
            "params_match": True,
            "validation_errors": [],
        },
    }

    function_generator = FunctionGenerator(code_generator=mock_code_generator)

    template_vars = function_generator.generate_function(
        requirements="Create a function that adds two numbers",
        function_name="test_function",
        parameters=[Parameter(name="x", type="int", description="Test parameter")],
        return_type="int",
        context="This is a test function",
    )

    assert template_vars is not None
    assert "function_name" in template_vars
    assert "parameters" in template_vars
    assert "return_type" in template_vars
    assert "requirements" in template_vars

    # Verify variables
    assert template_vars["function_name"] == "test_function"
    assert template_vars["return_type"] == "int"
    assert any("x: int" in param for param in template_vars["parameters"])
    assert "Create a function that adds two numbers" in template_vars["requirements"]
    assert "This is a test function" in template_vars["context"]
