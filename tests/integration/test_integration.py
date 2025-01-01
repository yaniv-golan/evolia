"""Integration tests for the Evolia system"""
import pytest
import os
from pathlib import Path
import json
import logging

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
def mock_responses():
    """Mock responses for OpenAI endpoints"""
    return {
        'generate_code': {
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
    }

def test_basic_execution(tmp_path):
    """Test basic execution of a plan"""
    config = {
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
    executor = Executor2(config, ephemeral_dir=str(tmp_path / "artifacts"))
    
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
    
    # Execute plan
    executor.execute_plan(plan)
    
    # Verify results
    assert executor.results['result'] == 8