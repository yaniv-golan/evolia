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
def complex_executor():
    """Create a RestrictedExecutor with real-world configuration."""
    allowed_modules = {
        'os',
        'os.path',
        'pathlib',
        'json',
        'pandas',
        'numpy',
        'datetime',
        'math',
        're',
        'random'
    }
    allowed_builtins = {
        'len', 'str', 'int', 'float', 'bool', 'dict', 'list', 'tuple',
        'print', 'sum', 'min', 'max', 'sorted', 'enumerate', 'zip',
        'round', 'abs', 'all', 'any', 'filter', 'map', 'range'
    }
    return RestrictedExecutor(allowed_modules, allowed_builtins)

def test_pandas_data_processing(complex_executor, tmp_path):
    """Test processing data with pandas in restricted environment."""
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
    
    # Read data
    df = pd.read_csv(inputs['data_file'])
    
    # Process data
    df['C'] = df['A'] + df['B']
    
    # Save results
    output_file = os.path.join(output_dir, 'result.csv')
    df.to_csv(output_file, index=False)
    
    return {
        'sum_C': float(df['C'].sum()),
        'output_file': output_file
    }
"""
    result = complex_executor.execute_in_sandbox(
        script,
        {'data_file': str(data_file)},
        str(tmp_path)
    )
    
    assert result['sum_C'] == 21.0
    assert os.path.exists(result['output_file'])
    
    # Verify output file
    df_result = pd.read_csv(result['output_file'])
    assert all(df_result['C'] == [5, 7, 9])

def test_numpy_computation(complex_executor, tmp_path):
    """Test numerical computations with numpy in restricted environment."""
    script = """
def main(inputs, output_dir):
    import numpy as np
    
    # Create and process array
    arr = np.array(inputs['data'])
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    
    # Save results
    result_file = os.path.join(output_dir, 'stats.json')
    with open(result_file, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    
    return {
        'mean': mean,
        'std': std,
        'result_file': result_file
    }
"""
    result = complex_executor.execute_in_sandbox(
        script,
        {'data': [1, 2, 3, 4, 5]},
        str(tmp_path)
    )
    
    assert result['mean'] == 3.0
    assert round(result['std'], 2) == 1.41
    
    # Verify saved results
    with open(result['result_file']) as f:
        saved = json.load(f)
    assert saved['mean'] == result['mean']
    assert saved['std'] == result['std']

def test_complex_data_transformation(complex_executor, tmp_path):
    """Test complex data transformation with multiple operations."""
    input_data = {
        'records': [
            {'id': 1, 'values': [1, 2, 3]},
            {'id': 2, 'values': [4, 5, 6]},
            {'id': 3, 'values': [7, 8, 9]}
        ]
    }
    
    script = """
def main(inputs, output_dir):
    import pandas as pd
    import numpy as np
    import json
    import os.path
    
    # Convert to DataFrame
    records = inputs['records']
    df = pd.DataFrame(records)
    
    # Process values
    df['mean'] = df['values'].apply(np.mean)
    df['sum'] = df['values'].apply(sum)
    df['stats'] = df['values'].apply(lambda x: {
        'min': min(x),
        'max': max(x),
        'len': len(x)
    })
    
    # Save detailed results
    output_file = os.path.join(output_dir, 'analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_records': len(df),
                'global_mean': float(df['mean'].mean()),
                'global_sum': float(df['sum'].sum())
            },
            'records': df.to_dict('records')
        }, f, indent=2)
    
    return {
        'record_count': len(df),
        'total_sum': float(df['sum'].sum()),
        'output_file': output_file
    }
"""
    result = complex_executor.execute_in_sandbox(script, input_data, str(tmp_path))
    
    assert result['record_count'] == 3
    assert result['total_sum'] == 45
    
    # Verify output file
    with open(result['output_file']) as f:
        analysis = json.load(f)
    assert analysis['summary']['total_records'] == 3
    assert analysis['summary']['global_sum'] == 45
    assert len(analysis['records']) == 3

def test_file_operations_security(complex_executor, tmp_path):
    """Test file operation security restrictions."""
    # Try to access file outside output directory
    script = """
def main(inputs, output_dir):
    import os.path
    
    # Attempt to write outside output directory
    with open('/tmp/test.txt', 'w') as f:
        f.write('test')
    
    return {'result': True}
"""
    with pytest.raises((RestrictedExecutionError, SecurityViolationError)):
        complex_executor.execute_in_sandbox(script, {}, str(tmp_path))

def test_network_access_restriction(complex_executor, tmp_path):
    """Test network access restrictions."""
    script = """
def main(inputs, output_dir):
    import socket  # Should fail
    
    # Try to make network connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('example.com', 80))
    
    return {'result': True}
"""
    with pytest.raises((RestrictedExecutionError, RestrictedImportError)):
        complex_executor.execute_in_sandbox(script, {}, str(tmp_path))

def test_resource_intensive_operation(complex_executor, tmp_path):
    """Test handling of resource-intensive operations."""
    script = """
def main(inputs, output_dir):
    import numpy as np
    
    # Create large array
    size = inputs['size']
    arr = np.random.rand(size, size)
    
    # Perform intensive computation
    result = np.linalg.svd(arr)
    
    return {
        'shape': arr.shape,
        'mean': float(np.mean(arr)),
        'success': True
    }
"""
    # Test with moderately large matrix
    result = complex_executor.execute_in_sandbox(
        script,
        {'size': 100},  # 100x100 matrix
        str(tmp_path)
    )
    
    assert result['shape'] == (100, 100)
    assert 0 <= result['mean'] <= 1  # Random values should be between 0 and 1
    assert result['success']

def test_concurrent_file_access(complex_executor, tmp_path):
    """Test concurrent file access handling."""
    # Create test files
    for i in range(3):
        with open(tmp_path / f"input_{i}.txt", "w") as f:
            f.write(f"content_{i}")
    
    script = """
def main(inputs, output_dir):
    import os
    import json
    
    results = {}
    
    # Read multiple files
    for i in range(3):
        input_file = os.path.join(output_dir, f'input_{i}.txt')
        with open(input_file) as f:
            results[f'content_{i}'] = f.read()
            
        # Write to output files
        output_file = os.path.join(output_dir, f'output_{i}.txt')
        with open(output_file, 'w') as f:
            f.write(f'processed_{results[f"content_{i}"]}')
    
    return {
        'input_contents': results,
        'files_processed': 3
    }
"""
    result = complex_executor.execute_in_sandbox(script, {}, str(tmp_path))
    
    assert result['files_processed'] == 3
    assert all(f"content_{i}" in result['input_contents'] for i in range(3))
    
    # Verify output files
    for i in range(3):
        output_file = tmp_path / f"output_{i}.txt"
        assert output_file.exists()
        assert output_file.read_text() == f"processed_content_{i}" 