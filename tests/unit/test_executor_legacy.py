"""Tests for the executor module"""
import pytest
import os
import sys
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from evolia.models import Plan, PlanStep
from evolia.core.executor import Executor
from evolia.utils.exceptions import CodeExecutionError, FileAccessViolationError

@pytest.fixture
def executor(tmp_path):
    """Create an executor instance for testing"""
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
    return Executor(config, keep_artifacts=True, ephemeral_dir=str(tmp_path / "artifacts"))

def test_file_access_control(executor, tmp_path):
    """Test file access control in execution"""
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

def test_write_outside_ephemeral(executor, tmp_path):
    """Test prevention of writing outside ephemeral directory"""
    # Create test plan that attempts to write outside ephemeral directory
    plan = Plan(steps=[
        PlanStep(
            name='Generate malicious function',
            tool='generate_code',
            inputs={
                'function_name': 'malicious_function',
                'parameters': [],
                'return_type': 'None'
            },
            outputs={'code_file': 'run_artifacts/step_1/malicious.py'},
            allowed_read_paths=[]
        ),
        PlanStep(
            name='Execute malicious function',
            tool='execute_code',
            inputs={
                'script_file': 'run_artifacts/step_1/malicious.py'
            },
            outputs={'result': 'result'},
            allowed_read_paths=['run_artifacts/step_1/malicious.py']
        )
    ])

    # Mock the code generation to create a file that attempts to write outside ephemeral
    with patch('evolia.executor.call_openai_structured') as mock_openai:
        mock_openai.return_value = {
            'code': '''
def main(inputs, output_dir):
    with open('/tmp/malicious.txt', 'w') as f:
        f.write('malicious')
    return {'result': 'success'}
''',
            'function_name': 'malicious_function',
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
        
        with pytest.raises(CodeExecutionError):
            executor.execute_plan(plan)

def test_path_traversal_prevention(executor, tmp_path):
    """Test prevention of path traversal attacks"""
    # Create test plan that attempts path traversal
    plan = Plan(steps=[
        PlanStep(
            name='Generate test function',
            tool='generate_code',
            inputs={
                'function_name': 'malicious_function',
                'parameters': [],
                'return_type': 'None'
            },
            outputs={'code_file': 'run_artifacts/step_1/malicious.py'},
            allowed_read_paths=[]
        ),
        PlanStep(
            name='Execute test function',
            tool='execute_code',
            inputs={
                'script_file': '../../etc/passwd'
            },
            outputs={'result': 'result'},
            allowed_read_paths=['../../etc/passwd']
        )
    ])

    # Mock the code generation to create a file that attempts path traversal
    with patch('evolia.executor.call_openai_structured') as mock_openai:
        mock_openai.return_value = {
            'code': '''
def main(inputs, output_dir):
    with open('../../etc/passwd', 'r') as f:
        data = f.read()
    return {'result': data}
''',
            'function_name': 'malicious_function',
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
        
        with pytest.raises(FileAccessViolationError):
            executor.execute_plan(plan)

def test_executor_loads_tools(executor):
    """Test that executor loads system tools correctly"""
    assert executor is not None
    assert executor.config is not None