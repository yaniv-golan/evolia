"""Unit tests for CodeGenerator."""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from evolia.core.code_generator import CodeGenerator, CodeGenerationConfig

@pytest.fixture
def mock_openai():
    """Mock OpenAI structured output calls."""
    with patch('evolia.core.code_generator.call_openai_structured') as mock:
        mock.return_value = {
            'code': 'def test(): pass',
            'validation_results': {
                'syntax_valid': True,
                'security_issues': []
            }
        }
        yield mock

@pytest.fixture
def code_generator():
    """Create a CodeGenerator instance with test config."""
    config = CodeGenerationConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=100,
        allowed_modules={'math'},
        allowed_builtins={'len', 'str'},
        api_key="test-key"
    )
    return CodeGenerator(config)

def test_code_generator_init_default_config():
    """Test CodeGenerator initialization with default config."""
    with pytest.raises(ValueError) as exc_info:
        CodeGenerator()
    assert "Configuration must be provided with API key" in str(exc_info.value)

def test_code_generator_init_custom_config():
    """Test CodeGenerator initialization with custom config."""
    config = CodeGenerationConfig(
        model="custom-model",
        temperature=0.7,
        max_tokens=500,
        api_key="test-key"
    )
    generator = CodeGenerator(config)
    assert generator.config.model == "custom-model"
    assert generator.config.temperature == 0.7
    assert generator.config.max_tokens == 500

def test_code_generator_config_validation():
    """Test CodeGenerationConfig validation."""
    with pytest.raises(ValueError):
        CodeGenerationConfig(temperature=1.5, api_key="test-key")
    
    with pytest.raises(ValueError):
        CodeGenerationConfig(top_p=1.5, api_key="test-key")
    
    with pytest.raises(ValueError):
        CodeGenerationConfig(frequency_penalty=3.0, api_key="test-key")
    
    with pytest.raises(ValueError):
        CodeGenerationConfig(presence_penalty=-3.0, api_key="test-key")
    
    with pytest.raises(ValueError):
        CodeGenerationConfig(max_tokens=0, api_key="test-key")
    
    with pytest.raises(ValueError):
        CodeGenerationConfig(api_key="")

def test_generate_success(code_generator, mock_openai):
    """Test successful code generation."""
    template = "Generate a function that {action}"
    vars = {"action": "adds two numbers"}
    schema = {"type": "object"}
    system_prompt = "You are a test assistant"
    
    response = code_generator.generate(
        prompt_template=template,
        template_vars=vars,
        schema=schema,
        system_prompt=system_prompt
    )
    
    assert response['code'] == 'def test(): pass'
    assert response['validation_results']['syntax_valid']
    mock_openai.assert_called_once()

def test_generate_with_temperature_override(code_generator, mock_openai):
    """Test code generation with temperature override."""
    template = "Test template"
    vars = {"test": "value"}
    schema = {"type": "object"}
    
    code_generator.generate(
        prompt_template=template,
        template_vars=vars,
        schema=schema,
        system_prompt="Test",
        temperature=0.8
    )
    
    call_args = mock_openai.call_args[1]
    assert call_args['temperature'] == 0.8

def test_generate_error_handling(code_generator, mock_openai):
    """Test error handling during generation."""
    mock_openai.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        code_generator.generate(
            prompt_template="test",
            template_vars={},
            schema={},
            system_prompt="test"
        )
    
    assert str(exc_info.value) == "Test error"

def test_generate_template_formatting(code_generator, mock_openai):
    """Test template formatting in generate."""
    template = "Create a function named {name} that {action}"
    vars = {
        "name": "test_func",
        "action": "does something"
    }
    
    code_generator.generate(
        prompt_template=template,
        template_vars=vars,
        schema={},
        system_prompt="test"
    )
    
    call_args = mock_openai.call_args[1]
    assert "test_func" in call_args['user_prompt']
    assert "does something" in call_args['user_prompt'] 