"""Tests for the Evolia core functionality"""
import pytest
import os
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from evolia.core.evolia import (
    generate_plan,
    execute_plan,
    validate_plan,
    PlanValidationError,
    PlanExecutionError,
    load_config,
    load_system_tools
)
from evolia.models import Plan, PlanStep, SystemTool
from evolia.utils.exceptions import CodeExecutionError
from evolia.integrations.openai_structured import call_openai_structured
from evolia.core.executor2 import Executor2

@pytest.fixture
def mock_config():
    """Test configuration"""
    return {
        'openai': {
            'model': 'gpt-4o-2024-08-06',
            'api_key_env_var': 'OPENAI_API_KEY',
            'max_retries': 3,
            'retry_delay': 1
        },
        'allowed_modules': ['math', 'random', 'typing'],
        'validation': {
            'max_lines': 100,
            'max_complexity': 10,
            'banned_functions': ['eval', 'exec', 'compile'],
            'max_syntax_lint_retries': 3,
            'max_runtime_retries': 3
        },
        'file_access': {
            'paths': {
                'ephemeral_base': 'test_artifacts',
                'tools_base': 'test_tools',
                'data_base': 'test_data'
            }
        }
    }

@pytest.fixture
def mock_system_tools():
    """Mock system tools for testing"""
    return [
        {
            'name': 'test_function',
            'description': 'A test function',
            'interface': {
                'function_name': 'test_function',
                'parameters': [
                    {'name': 'x', 'type': 'int', 'description': 'First number'},
                    {'name': 'y', 'type': 'int', 'description': 'Second number'}
                ],
                'return_type': 'int',
                'description': 'A test function that takes two numbers'
            },
            'metadata': {
                'loaded_at': '2024-12-30T20:35:18',
                'validated': True,
                'validation_errors': [],
                'last_execution': None,
                'execution_count': 0,
                'average_duration': 0.0,
                'success_rate': 0.0,
                'interface_version': '1.0',
                'permissions': {
                    'default_policy': 'deny',
                    'allow_read': [],
                    'allow_write': [],
                    'allow_create': []
                }
            }
        }
    ]

def test_generate_plan(mock_config, mock_system_tools, tmp_path):
    """Test plan generation"""
    task = "Test task"
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.allow_read = []
            self.allow_write = []
            self.allow_create = []
            self.default_policy = "deny"
            self.keep_artifacts = True
            self.ephemeral_dir = str(tmp_path / "artifacts")
    
    args = MockArgs()
    
    # Mock the OpenAI call
    with patch('evolia.core.evolia.call_openai_structured', autospec=True) as mock_openai:
        mock_openai.return_value = {
            'steps': [
                {
                    'name': 'Execute test function',
                    'tool': 'test_function',
                    'inputs': {
                        'x': 5,
                        'y': 3
                    },
                    'outputs': {'result': 'result'},
                    'allowed_read_paths': [],
                    'allowed_write_paths': [],
                    'allowed_create_paths': [],
                    'default_policy': 'deny'
                }
            ]
        }
        
        plan = generate_plan(task, mock_system_tools, mock_config, args)
        
        # Verify OpenAI was called with correct arguments
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args[1]
        assert call_args['api_key'] == os.environ.get('OPENAI_API_KEY')
        assert call_args['model'] == mock_config['openai']['model']
        assert 'json_schema' in call_args
        assert task in call_args['user_prompt']
        assert 'system_prompt' in call_args
        
        # Verify plan structure
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.name == 'Execute test function'
        assert step.tool == 'test_function'
        assert step.inputs == {'x': 5, 'y': 3}
        assert step.outputs == {'result': 'result'}
        assert step.allowed_read_paths == []
        assert step.allowed_write_paths == []
        assert step.allowed_create_paths == []
        assert step.default_policy == 'deny'

def test_generate_plan_error(mock_config, tmp_path):
    """Test error handling in plan generation"""
    task = "Test task"
    system_tools = []  # Empty tools list to trigger error
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.allow_read = []
            self.allow_write = []
            self.allow_create = []
            self.default_policy = "deny"
            self.keep_artifacts = True
            self.ephemeral_dir = str(tmp_path / "artifacts")
    
    args = MockArgs()
    
    # Mock OpenAI to return invalid response
    with patch('evolia.core.evolia.call_openai_structured', autospec=True) as mock_openai:
        mock_openai.return_value = {
            'steps': [
                {
                    'name': 'Invalid step',
                    'tool': 'nonexistent_tool',
                    'inputs': {},
                    'outputs': {},
                    'allowed_read_paths': [],
                    'allowed_write_paths': [],
                    'allowed_create_paths': [],
                    'default_policy': 'deny'
                }
            ]
        }
        
        plan = generate_plan(task, system_tools, mock_config, args)
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert not step.interface_validation.matches_interface
        assert any("Unknown tool" in error for error in step.interface_validation.validation_errors)

def test_load_config():
    """Test configuration loading"""
    mock_config = {
        'openai': {
            'model': 'gpt-4o-2024-08-06',
            'api_key_env_var': 'OPENAI_API_KEY',
            'max_retries': 3,
            'retry_delay': 1
        },
        'allowed_modules': ['math', 'random', 'typing'],
        'validation': {
            'max_lines': 100,
            'max_complexity': 10,
            'banned_functions': ['eval', 'exec', 'compile'],
            'max_syntax_lint_retries': 3,
            'max_runtime_retries': 3
        },
        'file_access': {
            'paths': {
                'ephemeral_base': 'test_artifacts',
                'tools_base': 'test_tools',
                'data_base': 'test_data'
            },
            'permissions': {
                'default_policy': 'deny',
                'allow_read': ['test_data'],
                'allow_write': ['test_artifacts'],
                'allow_create': ['test_artifacts']
            }
        }
    }
    
    with patch('yaml.safe_load', return_value=mock_config):
        config = load_config()
        
        # Verify config structure
        assert isinstance(config, dict)
        assert 'openai' in config
        assert config['openai']['model'] == 'gpt-4o-2024-08-06'
        assert config['openai']['api_key_env_var'] == 'OPENAI_API_KEY'
        assert config['openai']['max_retries'] == 3
        
        assert 'allowed_modules' in config
        assert 'math' in config['allowed_modules']
        assert 'random' in config['allowed_modules']
        
        assert 'validation' in config
        assert config['validation']['max_lines'] == 100
        assert config['validation']['max_complexity'] == 10
        assert 'eval' in config['validation']['banned_functions']
        
        assert 'file_access' in config
        assert 'paths' in config['file_access']
        assert config['file_access']['paths']['ephemeral_base'] == 'test_artifacts'
        assert 'permissions' in config['file_access']
        assert config['file_access']['permissions']['default_policy'] == 'deny'

def test_load_system_tools():
    """Test system tools loading"""
    mock_tool_content = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "interface": {
                "function_name": "test_function",
                "parameters": [
                    {"name": "input", "type": "str", "description": "Input value"}
                ],
                "return_type": "dict",
                "description": "A test function that processes input"
            },
            "metadata": {
                "loaded_at": "2024-12-30T20:35:18",
                "validated": True,
                "validation_errors": [],
                "last_execution": None,
                "execution_count": 0,
                "average_duration": 0.0,
                "success_rate": 0.0,
                "interface_version": "1.0",
                "permissions": {
                    "default_policy": "deny",
                    "allow_read": [],
                    "allow_write": [],
                    "allow_create": []
                }
            }
        }
    ]
    
    with patch('pathlib.Path.exists', return_value=True), \
         patch('json.load', return_value=mock_tool_content):
        
        tools = load_system_tools()
        
        # Verify tools were loaded
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Verify tool structure
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'interface' in tool
            assert 'metadata' in tool
            
            # Verify interface
            interface = tool['interface']
            assert 'function_name' in interface
            assert 'parameters' in interface
            assert 'return_type' in interface
            assert 'description' in interface
            
            # Verify metadata
            metadata = tool['metadata']
            assert 'loaded_at' in metadata
            assert 'validated' in metadata
            assert 'validation_errors' in metadata
            assert 'interface_version' in metadata
            assert 'permissions' in metadata

def test_prompt_for_promotion(monkeypatch):
    """Test promotion prompting"""
    test_file = 'test_file.py'
    
    # Test approval
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.stat') as mock_stat:
        mock_stat.return_value.st_size = 1024
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        
        result = prompt_for_promotion(test_file)
        assert result is True
    
    # Test rejection
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.stat') as mock_stat:
        mock_stat.return_value.st_size = 1024
        monkeypatch.setattr('builtins.input', lambda _: 'n')
        
        result = prompt_for_promotion(test_file)
        assert result is False
    
    # Test invalid input then valid input
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.stat') as mock_stat:
        mock_stat.return_value.st_size = 1024
        responses = iter(['invalid', 'y'])
        monkeypatch.setattr('builtins.input', lambda _: next(responses))
        
        result = prompt_for_promotion(test_file)
        assert result is True 