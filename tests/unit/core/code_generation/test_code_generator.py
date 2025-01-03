"""Unit tests for CodeGenerator."""

from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from evolia.core.code_generator import CodeGenerationConfig, CodeGenerator
from evolia.models.schemas import CODE_SCHEMA
from evolia.utils.exceptions import CodeGenerationError


@pytest.fixture
def mock_openai():
    """Mock OpenAI structured output calls."""
    with patch("evolia.core.code_generator.call_openai_structured") as mock:
        mock.return_value = {
            "code": '''def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return their sum.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        Sum of the two numbers
    """
    return a + b''',
            "validation_results": {"syntax_valid": True, "security_issues": []},
            "function_name": "add_numbers",
            "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            "return_type": "int",
            "description": "Add two numbers and return their sum.",
            "required_imports": [],
        }
        yield mock


@pytest.fixture
def code_generator(mock_openai_config):
    """Create a CodeGenerator instance with test config."""
    config = CodeGenerationConfig(**mock_openai_config)
    return CodeGenerator(config)


def test_code_generator_init(code_generator, mock_openai_config):
    """Test CodeGenerator initialization."""
    assert code_generator.config.model == mock_openai_config["model"]
    assert code_generator.config.temperature == mock_openai_config["temperature"]
    assert code_generator.config.max_tokens == mock_openai_config["max_tokens"]
    assert set(code_generator.config.allowed_modules) == set(
        mock_openai_config["allowed_modules"]
    )
    assert set(code_generator.config.allowed_builtins) == set(
        mock_openai_config["allowed_builtins"]
    )


def test_generate_success(code_generator, mock_openai):
    """Test successful code generation."""
    result = code_generator.generate(
        prompt_template="Generate a function that {action}",
        template_vars={"action": "adds two numbers"},
        schema=CODE_SCHEMA,
        system_prompt=None,
    )

    # Verify code structure
    assert "def add_numbers" in result["code"]
    assert "return a + b" in result["code"]
    assert '"""' in result["code"]  # Has docstring

    # Verify validation results
    assert result["validation_results"]["syntax_valid"]
    assert not result["validation_results"]["security_issues"]

    # Verify function metadata
    assert result["function_name"] == "add_numbers"
    assert len(result["parameters"]) == 2
    assert result["return_type"] == "int"
    assert result["description"]
    assert isinstance(result["required_imports"], list)


def test_generate_error_handling(code_generator, mock_openai):
    """Test error handling during code generation."""
    mock_openai.side_effect = Exception("Invalid response format")

    with pytest.raises(CodeGenerationError, match="Failed to generate code"):
        code_generator.generate(
            prompt_template="Generate a function that {action}",
            template_vars={"action": "adds two numbers"},
            schema=CODE_SCHEMA,
            system_prompt=None,
        )


def test_generate_with_overrides(code_generator, mock_openai):
    """Test code generation with config overrides."""
    result = code_generator.generate(
        prompt_template="Generate a function that {action}",
        template_vars={
            "action": "adds two numbers",
            "allowed_modules": ["math"],
            "allowed_builtins": ["sum"],
        },
        schema=CODE_SCHEMA,
        system_prompt="Custom system prompt",
    )

    assert result["code"]
    assert result["validation_results"]["syntax_valid"]


def test_generate_with_security_issues(code_generator, mock_openai):
    """Test handling of security issues in generated code."""
    mock_openai.return_value["validation_results"]["security_issues"] = [
        "Unsafe module import detected",
        "Potential code injection risk",
    ]

    result = code_generator.generate(
        prompt_template="Generate a function that {action}",
        template_vars={"action": "executes a shell command"},
        schema=CODE_SCHEMA,
        system_prompt=None,
    )

    assert result["validation_results"]["security_issues"]
    assert len(result["validation_results"]["security_issues"]) == 2
