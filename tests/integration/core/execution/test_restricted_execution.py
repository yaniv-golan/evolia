"""Integration tests for restricted execution functionality."""

import pytest
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from evolia.core.restricted_execution import (
    RestrictedExecutor,
    RestrictedExecutionError,
    RestrictedImportError,
    RestrictedAttributeError,
)
from evolia.utils.exceptions import SecurityViolationError


@pytest.fixture
def executor():
    """Create a RestrictedExecutor with essential configuration."""
    allowed_modules = {"os.path", "json", "pandas", "numpy", "math"}
    allowed_builtins = {
        "len",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "sum",
        "min",
        "max",
    }
    return RestrictedExecutor(
        allowed_modules=allowed_modules, allowed_builtins=allowed_builtins
    )


def test_data_processing(executor, tmp_path):
    """Test data processing in restricted environment."""
    # Create test data
    data_file = tmp_path / "data.csv"
    pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}).to_csv(data_file, index=False)

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
        script, {"data_file": str(data_file)}, str(tmp_path)
    )

    assert result["sum"] == 21.0
    assert os.path.exists(result["output_file"])

    df_result = pd.read_csv(result["output_file"])
    assert all(df_result["C"] == [5, 7, 9])


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


def test_parallel_file_access(executor, tmp_path):
    """Test parallel file access in restricted environment."""
    script = """
def main(inputs, output_dir):
    import os.path
    import json
    
    # Write to a unique file
    output_file = os.path.join(output_dir, f'output_{inputs["id"]}.json')
    data = {'id': inputs['id'], 'value': inputs['id'] * 2}
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    return {'file': output_file}
"""

    def run_with_id(id_num):
        return executor.execute_in_sandbox(script, {"id": id_num}, str(tmp_path))

    # Run multiple instances in parallel
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_with_id, i) for i in range(5)]
        results = [f.result() for f in futures]

    # Verify all files were created correctly
    for i in range(5):
        file_path = results[i]["file"]
        assert os.path.exists(file_path)
        with open(file_path) as f:
            data = json.load(f)
            assert data["id"] == i
            assert data["value"] == i * 2


def test_network_access(executor, tmp_path):
    """Test network access restrictions."""
    test_cases = [
        # Socket creation
        (
            """
def main(inputs, output_dir):
    import socket
    sock = socket.socket()
    return {'success': True}
""",
            "import of module 'socket' is not allowed",
        ),
        # HTTP request
        (
            """
def main(inputs, output_dir):
    import urllib.request
    response = urllib.request.urlopen('http://example.com')
    return {'success': True}
""",
            "import of module 'urllib.request' is not allowed",
        ),
        # High-level HTTP
        (
            """
def main(inputs, output_dir):
    import requests
    response = requests.get('http://example.com')
    return {'success': True}
""",
            "import of module 'requests' is not allowed",
        ),
    ]

    for script, expected_error in test_cases:
        with pytest.raises(
            (RestrictedExecutionError, RestrictedImportError)
        ) as exc_info:
            executor.execute_in_sandbox(script, {}, str(tmp_path))
        assert expected_error in str(exc_info.value).lower()
