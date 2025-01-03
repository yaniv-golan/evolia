"""Integration tests for execution output handling and processing."""
from unittest.mock import patch

import pytest

from evolia.core.code_generator import CodeGenerationConfig, CodeGenerator
from evolia.models import Parameter


@pytest.fixture
def code_generator():
    """Create a code generator for testing."""
    config = CodeGenerationConfig(model="gpt-4", api_key="test-key")
    return CodeGenerator(config)


@pytest.fixture
def mock_responses():
    """Mock responses for testing."""
    return {
        "step1": {
            "code": "def process_data(data: list) -> dict:\n    return {'count': len(data)}",
            "function_name": "process_data",
            "parameters": [{"name": "data", "type": "list"}],
            "return_type": "dict",
            "validation_results": {"syntax_valid": True, "security_issues": []},
        },
        "step2": {
            "code": "def analyze_result(result: dict) -> int:\n    return result['count']",
            "function_name": "analyze_result",
            "parameters": [{"name": "result", "type": "dict"}],
            "return_type": "int",
            "validation_results": {"syntax_valid": True, "security_issues": []},
        },
    }


def test_basic_output_handling(code_generator, mock_responses):
    """Test basic output handling."""
    with patch("evolia.core.code_generator.call_openai_structured") as mock_openai:
        mock_openai.return_value = mock_responses["step1"]

        template_vars = {
            "description": "Process input data",
            "parameters": [
                Parameter(name="data", type="list", description="Input data")
            ],
            "return_type": "dict",
        }

        response = code_generator.generate(
            prompt_template="Generate a function to {description} with parameters {parameters} that returns a {return_type}",
            template_vars=template_vars,
            schema={"type": "object", "properties": {}},
        )

        assert response is not None
        assert "code" in response
        assert response["function_name"] == "process_data"


def test_output_references(code_generator, mock_responses):
    """Test handling of output references."""
    with patch("evolia.core.code_generator.call_openai_structured") as mock_openai:
        mock_openai.return_value = mock_responses["step1"]

        template_vars = {
            "description": "Process input data",
            "parameters": [
                Parameter(name="data", type="list", description="Input data")
            ],
            "return_type": "dict",
        }

        step1_response = code_generator.generate(
            prompt_template="Generate a function to {description} with parameters {parameters} that returns a {return_type}",
            template_vars=template_vars,
            schema={"type": "object", "properties": {}},
        )

        mock_openai.return_value = mock_responses["step2"]

        template_vars = {
            "description": "Transform output",
            "parameters": [
                Parameter(name="data", type="dict", description="Input data")
            ],
            "return_type": "list",
        }

        step2_response = code_generator.generate(
            prompt_template="Generate a function to {description} with parameters {parameters} that returns a {return_type}",
            template_vars=template_vars,
            schema={"type": "object", "properties": {}},
        )

        assert step1_response is not None
        assert step2_response is not None
        assert "code" in step1_response
        assert "code" in step2_response
        assert step1_response["return_type"] == step2_response["parameters"][0]["type"]
