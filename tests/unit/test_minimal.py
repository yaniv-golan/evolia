"""Tests for minimal execution functionality"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from evolia.models import Plan, PlanStep
from evolia.core.executor2 import Executor2
from evolia.utils.exceptions import CodeExecutionError

@pytest.fixture
def config():
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
        }
    }

@pytest.fixture
def mock_openai_response():
    """Mock response for OpenAI endpoint calls"""
    def _mock_response(*args, **kwargs):
        return {
            'function_name': 'test_function',
            'parameters': [
                {'name': 'x', 'type': 'int'},
                {'name': 'y', 'type': 'int'}
            ],
            'return_type': 'int',
            'code': 'def test_function(x: int, y: int) -> int:\n    return x + y',
            'validation_results': {
                'syntax_valid': True,
                'name_matches': True,
                'params_match': True,
                'return_type_matches': True,
                'security_issues': []
            }
        }
    return _mock_response

def test_minimal_execution(config, mock_openai_response, tmp_path):
    """Test minimal execution flow"""
    # Create test plan
    plan = Plan(steps=[
        PlanStep(
            name='Generate test function',
            tool='generate_code',
            inputs={
                'function_name': 'test_function',
                'parameters': [
                    {'name': 'x', 'type': 'int'},
                    {'name': 'y', 'type': 'int'}
                ],
                'return_type': 'int'
            },
            outputs={'code_file': 'run_artifacts/step_1/test_function.py'},
            allowed_read_paths=[]
        ),
        PlanStep(
            name='Execute test function',
            tool='execute_code',
            inputs={
                'script_file': 'run_artifacts/step_1/test_function.py',
                'x': 5,
                'y': 3
            },
            outputs={'result': 'result'},
            allowed_read_paths=['run_artifacts/step_1/test_function.py']
        )
    ])
    
    with patch('evolia.openai_structured.call_openai_structured') as mock_openai:
        mock_openai.side_effect = mock_openai_response
        
        # Execute plan
        executor = Executor2(config, keep_artifacts=True, ephemeral_dir=str(tmp_path / "artifacts"))
        executor.execute_plan(plan)
        
        # Verify results
        assert executor.results['result'] == 8

def test_error_handling(config, mock_openai_response, tmp_path):
    """Test error handling in minimal execution"""
    # Create test plan with code that will fail during execution
    plan = Plan(steps=[
        PlanStep(
            name='Generate invalid function',
            tool='generate_code',
            inputs={
                'function_name': 'invalid_function',
                'parameters': [],
                'return_type': 'None'
            },
            outputs={'code_file': 'run_artifacts/step_1/invalid_function.py'},
            allowed_read_paths=[]
        ),
        PlanStep(
            name='Execute invalid function',
            tool='execute_code',
            inputs={
                'script_file': 'run_artifacts/step_1/invalid_function.py'
            },
            outputs={'result': 'result'},
            allowed_read_paths=['run_artifacts/step_1/invalid_function.py']
        )
    ])
    
    with patch('evolia.openai_structured.call_openai_structured') as mock_openai:
        mock_openai.return_value = {
            'code': '''
def main(inputs, output_dir):
    raise RuntimeError("Test execution error")
    return {'result': 'success'}
''',
            'function_name': 'invalid_function',
            'parameters': [],
            'return_type': 'None',
            'validation_results': {
                'syntax_valid': True,
                'name_matches': True,
                'params_match': True,
                'return_type_matches': True,
                'security_issues': []
            }
        }
        
        # Execute plan and verify it fails
        executor = Executor2(config, keep_artifacts=True, ephemeral_dir=str(tmp_path / "artifacts"))
        with pytest.raises(CodeExecutionError):
            executor.execute_plan(plan)