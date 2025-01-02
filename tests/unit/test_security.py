"""Unit tests for security module."""

import pytest
import ast
from typing import Dict, Any

from evolia.security.security import (
    SecurityVisitor,
    SecurityViolationError,
    validate_security_checks,
    get_subprocess_policy,
    validate_command,
    validate_code_security
)

@pytest.fixture
def security_config():
    """Create a security configuration for testing."""
    return {
        "security_checks": ["imports", "subprocess", "system_calls", "globals", "nonlocals", "nested_functions"],
        "allowed_modules": ["os", "pathlib", "json"],
        "allowed_builtins": ["len", "str", "int", "float"],
        "subprocess_policy": {
            "level": "none",
            "allowed_commands": [],
            "blocked_commands": []
        }
    }

@pytest.fixture
def security_visitor(security_config):
    """Create a security visitor for testing."""
    return SecurityVisitor(security_config)

def test_validate_nonlocals():
    """Test validation of nonlocal variables."""
    code = """
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
    return inner
"""
    config = {
        "security_checks": ["nonlocals"],
        "subprocess_policy": {"level": "none"}
    }
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, config)
    assert "nonlocal" in str(exc.value.violations.get("nonlocals", []))

def test_validate_safe_code(security_visitor):
    """Test validation of safe code."""
    code = """
def safe_function(x: int, y: int) -> int:
    return x + y
"""
    node = ast.parse(code)
    security_visitor.visit(node)  # Should not raise

def test_subprocess_none_policy():
    """Test subprocess none policy."""
    code = """
import subprocess
subprocess.run(["echo", "test"])
"""
    config = {
        "security_checks": ["subprocess"],
        "subprocess_policy": {"level": "none"}
    }
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, config)
    assert "not allowed under default policy" in str(exc.value.violations.get("subprocess", []))

def test_subprocess_always_policy():
    """Test subprocess always policy."""
    code = """
import subprocess
subprocess.run(["echo", "test"])
"""
    config = {
        "security_checks": ["subprocess"],
        "subprocess_policy": {
            "level": "always",
            "allowed_commands": ["echo"]
        }
    }
    validate_code_security(code, config)  # Should not raise

def test_subprocess_blocked_command():
    """Test blocked command handling."""
    code = """
import subprocess
subprocess.run(["rm", "-rf", "/"])
"""
    config = {
        "security_checks": ["subprocess"],
        "subprocess_policy": {
            "level": "always",
            "blocked_commands": ["rm"]
        }
    }
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, config)
    assert "blocked command" in str(exc.value.violations.get("subprocess", []))

def test_subprocess_system_tool_policy():
    """Test system tool policy."""
    code = """
import subprocess
subprocess.run(["git", "status"])
"""
    config = {
        "security_checks": ["subprocess"],
        "subprocess_policy": {
            "level": "system_tool",
            "allowed_commands": ["git"]
        }
    }
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, config)
    assert "system tool" in str(exc.value.violations.get("subprocess", []))

def test_subprocess_rate_limit():
    """Test rate limiting."""
    code = """
import subprocess
subprocess.run(["echo", "test"])
subprocess.run(["echo", "test2"])
"""
    config = {
        "security_checks": ["subprocess"],
        "subprocess_policy": {
            "level": "always",
            "rate_limit": {
                "enabled": True,
                "max_calls": 1,
                "period_seconds": 60
            }
        }
    }
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, config)
    assert "rate limit" in str(exc.value.violations.get("subprocess", []))

def test_subprocess_command_validation():
    """Test command validation."""
    config = {
        "security_checks": ["subprocess"],
        "subprocess_policy": {
            "level": "always",
            "blocked_commands": ["rm"]
        }
    }
    
    visitor = SecurityVisitor(config)
    node = ast.parse('subprocess.run(["rm", "-rf", "/"])')
    with pytest.raises(SecurityViolationError) as exc:
        visitor.visit(node)
    assert "blocked command" in str(exc.value.violations.get("subprocess", []))
    
    node = ast.parse('subprocess.run("echo test | grep foo", shell=True)')
    with pytest.raises(SecurityViolationError) as exc:
        visitor.visit(node)
    assert "shell" in str(exc.value.violations.get("subprocess", []))

def test_subprocess_logging(security_visitor, caplog):
    """Test subprocess logging."""
    code = """
import subprocess
subprocess.run(["echo", "test"])
"""
    node = ast.parse(code)
    try:
        security_visitor.visit(node)
    except SecurityViolationError:
        pass
    
    assert any("Subprocess call denied" in record.message for record in caplog.records)