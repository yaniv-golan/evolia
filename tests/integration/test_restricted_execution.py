"""Integration tests for restricted execution functionality."""

import pytest
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from evolia.core.restricted_execution import (
    RestrictedExecutor,
    RestrictedExecutionError,
    RestrictedImportError,
    RestrictedAttributeError
)
from evolia.utils.exceptions import SecurityViolationError

@pytest.fixture
def executor():
    """Create a RestrictedExecutor with essential configuration."""
    allowed_modules = {
        'os.path',
        'json',
        'pandas',
        'numpy',
        'math'
    }
    allowed_builtins = {
        'len', 'str', 'int', 'float', 'list', 'dict',
        'sum', 'min', 'max'
    }
    return RestrictedExecutor(allowed_modules, allowed_builtins)

def test_data_processing(executor, tmp_path):
    """Test data processing in restricted environment."""
    # Create test data
    data_file = tmp_path / "data.csv"
    pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }).to_csv(data_file, index=False)
    
    script = """
def main(inputs, output_dir):
    import pandas as pd
    import os.path
    
    df = pd.read_csv(inputs['data_file'])
    df['C'] = df['A'] + df['B']
    
    output_file = os.path.join(output_dir, 'result.csv')
    df.to_csv(output_file, index=False)
    
    return {
        'sum': float(df['C'].sum()),
        'output_file': output_file
    }
"""
    result = executor.execute_in_sandbox(
        script,
        {'data_file': str(data_file)},
        str(tmp_path)
    )
    
    assert result['sum'] == 21.0
    assert os.path.exists(result['output_file'])
    
    df_result = pd.read_csv(result['output_file'])
    assert all(df_result['C'] == [5, 7, 9])

def test_computation(executor, tmp_path):
    """Test numerical computations in restricted environment."""
    script = """
def main(inputs, output_dir):
    import numpy as np
    import json
    import os.path
    
    arr = np.array(inputs['data'])
    stats = {
        'mean': float(np.mean(arr)),
        'sum': float(np.sum(arr))
    }
    
    output_file = os.path.join(output_dir, 'stats.json')
    with open(output_file, 'w') as f:
        json.dump(stats, f)
    
    return stats
"""
    result = executor.execute_in_sandbox(
        script,
        {'data': [1, 2, 3, 4, 5]},
        str(tmp_path)
    )
    
    assert result['mean'] == 3.0
    assert result['sum'] == 15.0

def test_security_restrictions(executor, tmp_path):
    """Test security restrictions."""
    # Test file access restriction
    script1 = """
def main(inputs, output_dir):
    with open('/etc/passwd', 'r') as f:
        return {'content': f.read()}
"""
    with pytest.raises((RestrictedExecutionError, SecurityViolationError)):
        executor.execute_in_sandbox(script1, {}, str(tmp_path))
    
    # Test import restriction
    script2 = """
def main(inputs, output_dir):
    import socket
    return {'success': True}
"""
    with pytest.raises((RestrictedExecutionError, RestrictedImportError)):
        executor.execute_in_sandbox(script2, {}, str(tmp_path))
    
    # Test builtin restriction
    script3 = """
def main(inputs, output_dir):
    eval('print("hello")')
    return {'success': True}
"""
    with pytest.raises((RestrictedExecutionError, RestrictedAttributeError)):
        executor.execute_in_sandbox(script3, {}, str(tmp_path)) 