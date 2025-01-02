"""Unit tests for CodeGenerator."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig
from evolia.utils.exceptions import CodeGenerationError

@pytest.fixture
def mock_openai():
    """Mock OpenAI structured output calls."""
    with patch('evolia.core.code_generator.call_openai_structured') as mock:
        mock.return_value = {
            'code': '''def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return their sum.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        Sum of the two numbers
    """
    return a + b''',
            'validation_results': {
                'syntax_valid': True,
                'security_issues': [],
                'complexity_score': 1,
                'type_check_passed': True
            },
            'function_info': {
                'name': 'add_numbers',
                'parameters': [
                    {'name': 'a', 'type': 'int'},
                    {'name': 'b', 'type': 'int'}
                ],
                'return_type': 'int',
                'docstring': 'Add two numbers and return their sum.'
            }
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
    assert set(code_generator.config.allowed_modules) == set(mock_openai_config["allowed_modules"])
    assert set(code_generator.config.allowed_builtins) == set(mock_openai_config["allowed_builtins"])

def test_generate_success(code_generator, mock_openai):
    """Test successful code generation."""
    result = code_generator.generate(
        prompt_template="Generate a function that {action}",
        template_vars={"action": "adds two numbers"},
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "validation_results": {
                    "type": "object",
                    "properties": {
                        "syntax_valid": {"type": "boolean"},
                        "security_issues": {"type": "array"}
                    }
                }
            }
        }
    )
    
    # Verify code structure
    assert 'def add_numbers' in result['code']
    assert 'return a + b' in result['code']
    assert '"""' in result['code']  # Has docstring
    
    # Verify validation results
    assert result['validation_results']['syntax_valid']
    assert not result['validation_results']['security_issues']
    assert result['validation_results']['type_check_passed']
    assert result['validation_results']['complexity_score'] == 1
    
    # Verify function info
    assert result['function_info']['name'] == 'add_numbers'
    assert len(result['function_info']['parameters']) == 2
    assert result['function_info']['return_type'] == 'int'
    assert result['function_info']['docstring']
    
    # Verify OpenAI call
    mock_openai.assert_called_once()

def test_generate_error_handling(code_generator, mock_openai):
    """Test error handling during generation."""
    # Test API error
    mock_openai.side_effect = Exception("API Error")
    with pytest.raises(CodeGenerationError, match="Failed to generate code: API Error"):
        code_generator.generate(
            prompt_template="Invalid prompt",
            template_vars={},
            schema={}
        )
    
    # Test invalid response format
    mock_openai.side_effect = None
    mock_openai.return_value = {'invalid': 'response'}
    with pytest.raises(CodeGenerationError, match="Invalid response format"):
        code_generator.generate(
            prompt_template="Test prompt",
            template_vars={},
            schema={}
        )

def test_generate_with_overrides(code_generator, mock_openai):
    """Test generation with parameter overrides."""
    result = code_generator.generate(
        prompt_template="Test prompt",
        template_vars={},
        schema={},
        temperature=0.8,
        max_tokens=200,
        allowed_modules={'numpy', 'pandas'},
        allowed_builtins={'print', 'sum'}
    )
    
    call_args = mock_openai.call_args[1]
    assert call_args['temperature'] == 0.8
    assert call_args['max_tokens'] == 200
    
    # Original config should be unchanged
    assert code_generator.config.temperature == 0.5
    assert code_generator.config.max_tokens == 100
    assert code_generator.config.allowed_modules == {'math', 'typing'}
    assert code_generator.config.allowed_builtins == {'len', 'str', 'int', 'float'}

def test_generate_with_security_issues(code_generator, mock_openai):
    """Test handling of security issues in generated code."""
    mock_openai.return_value['validation_results']['security_issues'] = [
        "Use of eval() detected",
        "Potential shell injection"
    ]
    
    with pytest.raises(CodeGenerationError, match="Security issues detected"):
        code_generator.generate(
            prompt_template="Test prompt",
            template_vars={},
            schema={}
        ) 