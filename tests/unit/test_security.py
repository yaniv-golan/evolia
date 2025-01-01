"""Tests for security functionality"""
import pytest
from evolia.security.security import validate_code_security, SecurityVisitor
from evolia.utils.exceptions import SecurityViolationError
import ast

@pytest.fixture
def security_config():
    """Test security configuration"""
    return {
        'allowed_modules': ['requests', 'pandas', 'json', 'csv', 'os', 'pathlib'],
        'security_checks': [
            'no_system_calls',
            'no_file_operations',
            'no_network_access',
            'no_eval_exec',
            'no_subprocess'
        ]
    }

def test_validate_unsafe_imports(security_config):
    """Test validation of unsafe imports"""
    code = """import os
import subprocess
import sys

def test():
    pass"""
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, security_config)
    assert 'subprocess' in str(exc.value)

def test_validate_eval_exec(security_config):
    """Test validation of eval/exec usage"""
    code = """def test():
    x = eval('2 + 2')
    exec('print("Hello")')"""
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, security_config)
    assert 'eval' in str(exc.value)
    assert 'exec' in str(exc.value)

def test_validate_system_calls(security_config):
    """Test validation of system calls"""
    code = """import os

def test():
    os.system('rm -rf /')"""
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, security_config)
    assert 'system' in str(exc.value)

def test_validate_nested_functions(security_config):
    """Test validation of nested function definitions"""
    code = """def outer():
    def inner():
        pass
    return inner"""
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, security_config)
    assert 'nested' in str(exc.value)

def test_validate_globals(security_config):
    """Test validation of global statements"""
    code = """x = 1

def test():
    global x
    x = 2"""
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, security_config)
    assert 'global' in str(exc.value)

def test_validate_nonlocals():
    """Test validation of nonlocal statements"""
    code = """
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
    return inner
"""
    config = {
        'allowed_modules': ['requests', 'pandas', 'json', 'csv', 'os', 'pathlib'],
        'security_checks': ['no_nonlocals']
    }
    
    with pytest.raises(SecurityViolationError) as exc:
        validate_code_security(code, config)
    
    assert 'nested_functions' in exc.value.violations
    assert any('Nested function definition' in msg for msg in exc.value.violations['nested_functions'])

def test_validate_safe_code(security_config):
    """Test validation of safe code"""
    code = """def add(a: int, b: int) -> int:
    return a + b"""
    result = validate_code_security(code, security_config)
    assert result == {}

def create_config(policy_level="none", allowed_commands=None, blocked_commands=None, rate_limit=None):
    """Helper function to create test config"""
    config = {
        "subprocess_policy": {
            "level": policy_level,
            "allowed_commands": allowed_commands or [],
            "blocked_commands": blocked_commands or [],
            "rate_limit": rate_limit or {"enabled": False}
        }
    }
    return config

def create_subprocess_node(command):
    """Helper function to create a subprocess call AST node"""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='subprocess', ctx=ast.Load()),
            attr='run',
            ctx=ast.Load()
        ),
        args=[ast.Constant(value=command)],
        keywords=[]
    )

def test_subprocess_none_policy():
    """Test that 'none' policy blocks all subprocess calls"""
    config = create_config(policy_level="none")
    visitor = SecurityVisitor(config)
    node = create_subprocess_node("echo test")
    
    with pytest.raises(SecurityViolationError, match="not allowed under none policy"):
        visitor.check_subprocess_call(node, "echo test")

def test_subprocess_always_policy():
    """Test that 'always' policy allows subprocess calls"""
    config = create_config(policy_level="always")
    visitor = SecurityVisitor(config)
    node = create_subprocess_node("echo test")
    
    # Should not raise
    visitor.check_subprocess_call(node, "echo test")

def test_subprocess_blocked_command():
    """Test that blocked commands are not allowed regardless of policy"""
    config = create_config(
        policy_level="always",
        blocked_commands=["rm", "sudo"]
    )
    visitor = SecurityVisitor(config)
    node = create_subprocess_node("sudo ls")
    
    with pytest.raises(SecurityViolationError, match="is blocked by policy"):
        visitor.check_subprocess_call(node, "sudo ls")

def test_subprocess_system_tool_policy():
    """Test system_tool policy"""
    config = create_config(
        policy_level="system_tool",
        allowed_commands=["git", "pip"]
    )
    
    # Test with non-system tool
    visitor = SecurityVisitor(config, invoked_by_tool=False)
    node = create_subprocess_node("ls")
    with pytest.raises(SecurityViolationError, match="not an allowed system tool"):
        visitor.check_subprocess_call(node, "ls")
    
    # Test with system tool
    visitor = SecurityVisitor(config, invoked_by_tool=True)
    node = create_subprocess_node("git status")
    visitor.check_subprocess_call(node, "git status")  # Should not raise

def test_subprocess_rate_limit():
    """Test rate limiting for subprocess calls"""
    config = create_config(
        policy_level="always",
        rate_limit={
            "enabled": True,
            "max_calls": 2,
            "period_seconds": 60
        }
    )
    visitor = SecurityVisitor(config)
    node = create_subprocess_node("echo test")
    
    # First two calls should succeed
    visitor.check_subprocess_call(node, "echo test")
    visitor.check_subprocess_call(node, "echo test")
    
    # Third call should fail
    with pytest.raises(SecurityViolationError, match="rate limit exceeded"):
        visitor.check_subprocess_call(node, "echo test")

def test_subprocess_invalid_policy():
    """Test invalid policy level"""
    config = create_config(policy_level="invalid")
    visitor = SecurityVisitor(config)
    node = create_subprocess_node("echo test")
    
    with pytest.raises(SecurityViolationError, match="Invalid subprocess policy level"):
        visitor.check_subprocess_call(node, "echo test")

def test_subprocess_command_validation():
    """Test command validation rules"""
    config = create_config(policy_level="always")
    visitor = SecurityVisitor(config)
    
    # Test shell metacharacters
    with pytest.raises(SecurityViolationError, match="shell metacharacters"):
        visitor.validate_command("echo test | grep foo")
    
    # Test command length
    long_command = "x" * 1001
    with pytest.raises(SecurityViolationError, match="length exceeds security limit"):
        visitor.validate_command(long_command)
    
    # Test suspicious patterns
    with pytest.raises(SecurityViolationError, match="suspicious pattern"):
        visitor.validate_command("rm -rf /")

def test_subprocess_allowed_commands():
    """Test allowed commands list"""
    config = create_config(
        policy_level="system_tool",
        allowed_commands=["git", "pip"]
    )
    visitor = SecurityVisitor(config, invoked_by_tool=True)
    
    # Test allowed command
    node = create_subprocess_node("git status")
    visitor.check_subprocess_call(node, "git status")  # Should not raise
    
    # Test command with allowed prefix
    node = create_subprocess_node("git push origin main")
    visitor.check_subprocess_call(node, "git push origin main")  # Should not raise

def test_subprocess_invoked_by_tool():
    """Test invoked_by_tool flag behavior"""
    config = create_config(
        policy_level="system_tool",
        allowed_commands=["git"]
    )
    
    # Test without invoked_by_tool
    visitor = SecurityVisitor(config, invoked_by_tool=False)
    node = create_subprocess_node("echo test")
    with pytest.raises(SecurityViolationError):
        visitor.check_subprocess_call(node, "echo test")
    
    # Test with invoked_by_tool
    visitor = SecurityVisitor(config, invoked_by_tool=True)
    node = create_subprocess_node("echo test")
    visitor.check_subprocess_call(node, "echo test")  # Should not raise

def test_subprocess_logging(caplog):
    """Test logging of subprocess calls"""
    config = create_config(policy_level="always")
    visitor = SecurityVisitor(config)
    node = create_subprocess_node("echo test")
    
    visitor.check_subprocess_call(node, "echo test")
    
    # Check that appropriate log messages were generated
    assert any("Checking subprocess call" in record.message for record in caplog.records)
    assert any("Subprocess call allowed" in record.message for record in caplog.records)