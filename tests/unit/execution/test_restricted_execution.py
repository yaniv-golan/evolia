"""Unit tests for restricted execution functionality."""

import os
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from evolia.core.restricted_execution import (
    RestrictedAttributeError,
    RestrictedExecutionError,
    RestrictedExecutor,
    RestrictedImportError,
    guarded_getitem,
    restricted_import,
)


@pytest.fixture
def restricted_executor():
    """Create a RestrictedExecutor instance for testing."""
    allowed_modules = {"os.path", "pathlib", "json", "math"}
    allowed_builtins = {"len", "str", "int", "dict", "list", "print"}
    return RestrictedExecutor(allowed_modules, allowed_builtins)


def test_guarded_getitem_allowed_types():
    """Test guarded_getitem with allowed types."""
    # Test dictionary access
    d = {"key": "value"}
    assert guarded_getitem(d, "key") == "value"

    # Test list access
    l = [1, 2, 3]
    assert guarded_getitem(l, 1) == 2

    # Test tuple access
    t = (1, 2, 3)
    assert guarded_getitem(t, 1) == 2

    # Test string access
    s = "hello"
    assert guarded_getitem(s, 1) == "e"


def test_guarded_getitem_forbidden_types():
    """Test guarded_getitem with forbidden types."""

    class CustomObject:
        def __getitem__(self, key):
            return key

    obj = CustomObject()
    with pytest.raises(TypeError):
        guarded_getitem(obj, "any_key")


def test_restricted_import_allowed():
    """Test restricted_import with allowed modules."""
    allowed_modules = {"os.path", "pathlib", "json"}

    # Test importing os.path
    module = restricted_import("os.path", allowed_modules=allowed_modules)
    assert module is not None
    assert hasattr(module, "join")  # os.path.join should exist

    # Test importing pathlib
    module = restricted_import("pathlib", allowed_modules=allowed_modules)
    assert module is not None
    assert hasattr(module, "Path")  # pathlib.Path should exist

    # Test importing json
    module = restricted_import("json", allowed_modules=allowed_modules)
    assert module is not None
    assert hasattr(module, "dumps")  # json.dumps should exist


def test_restricted_import_forbidden():
    """Test restricted_import with forbidden modules."""
    allowed_modules = {"os.path", "pathlib"}

    # Test importing forbidden module
    with pytest.raises(RestrictedImportError):
        restricted_import("sys", allowed_modules=allowed_modules)


def test_prepare_restricted_globals(restricted_executor, tmp_path):
    """Test preparation of restricted globals."""
    inputs = {"param1": "value1"}
    output_dir = str(tmp_path)

    globals_dict = restricted_executor.prepare_restricted_globals(inputs, output_dir)

    # Check basic structure
    assert "__builtins__" in globals_dict
    assert "_getiter_" in globals_dict
    assert "_getitem_" in globals_dict
    assert "inputs" in globals_dict
    assert "output_dir" in globals_dict

    # Check inputs are properly set
    assert globals_dict["inputs"] == inputs
    assert globals_dict["output_dir"] == output_dir

    # Check allowed modules are imported
    for module in restricted_executor.allowed_modules:
        if "." not in module:  # Skip submodules for this test
            assert module in globals_dict


def test_execute_in_sandbox_success(restricted_executor, tmp_path):
    """Test successful code execution in sandbox."""
    script = """
def main(inputs, output_dir):
    return {'result': len(inputs)}
"""
    inputs = {"test": "value"}
    result = restricted_executor.execute_in_sandbox(script, inputs, str(tmp_path))

    assert isinstance(result, dict)
    assert result["result"] == 1


def test_execute_in_sandbox_syntax_error(restricted_executor, tmp_path):
    """Test handling of syntax errors in sandbox."""
    script = """
def main(inputs output_dir):  # Missing comma
    return {'result': True}
"""
    with pytest.raises(RestrictedExecutionError):
        restricted_executor.execute_in_sandbox(script, {}, str(tmp_path))


def test_execute_in_sandbox_runtime_error(restricted_executor, tmp_path):
    """Test handling of runtime errors in sandbox."""
    script = """
def main(inputs, output_dir):
    return {'result': undefined_variable}  # Reference undefined variable
"""
    with pytest.raises(RestrictedExecutionError):
        restricted_executor.execute_in_sandbox(script, {}, str(tmp_path))


def test_execute_in_sandbox_forbidden_import(restricted_executor, tmp_path):
    """Test handling of forbidden imports in sandbox."""
    script = """
def main(inputs, output_dir):
    import sys  # Forbidden import
    return {'result': True}
"""
    with pytest.raises((RestrictedExecutionError, RestrictedImportError)):
        restricted_executor.execute_in_sandbox(script, {}, str(tmp_path))


def test_execute_in_sandbox_file_operations(restricted_executor, tmp_path):
    """Test file operations in sandbox."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    script = """
def main(inputs, output_dir):
    with open(inputs['file_path'], 'r') as f:
        content = f.read()
    return {'result': content}
"""
    result = restricted_executor.execute_in_sandbox(
        script, {"file_path": str(test_file)}, str(tmp_path)
    )

    assert result["result"] == "test content"


def test_execute_in_sandbox_invalid_return(restricted_executor, tmp_path):
    """Test handling of invalid return values in sandbox."""
    script = """
def main(inputs, output_dir):
    return 'not a dict'  # Invalid return type
"""
    with pytest.raises(RestrictedExecutionError):
        restricted_executor.execute_in_sandbox(script, {}, str(tmp_path))


def test_execute_in_sandbox_missing_main(restricted_executor, tmp_path):
    """Test handling of missing main function in sandbox."""
    script = """
def helper():
    return {'result': True}
"""
    with pytest.raises(RestrictedExecutionError):
        restricted_executor.execute_in_sandbox(script, {}, str(tmp_path))


def test_execute_in_sandbox_allowed_builtins(restricted_executor, tmp_path):
    """Test access to allowed builtins in sandbox."""
    script = """
def main(inputs, output_dir):
    # Use various allowed builtins
    return {
        'str_result': str(123),
        'int_result': int('456'),
        'list_result': len([1, 2, 3])
    }
"""
    result = restricted_executor.execute_in_sandbox(script, {}, str(tmp_path))

    assert result["str_result"] == "123"
    assert result["int_result"] == 456
    assert result["list_result"] == 3
