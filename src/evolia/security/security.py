"""Security module for code validation and execution."""

import ast
import logging
import time
import shlex
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path

from ..utils.exceptions import SecurityViolationError, FileAccessViolationError
from ..utils.logger import setup_logger
from ..core.library_management import LibraryManager

logger = setup_logger()

# Define available security checks
AVAILABLE_SECURITY_CHECKS = {
    'imports': 'Check for unauthorized imports',
    'subprocess': 'Check subprocess calls',
    'system_calls': 'Check for system/popen calls',
    'globals': 'Check for global statements',
    'nonlocals': 'Check for nonlocal statements',
    'nested_functions': 'Check for nested function definitions',
    'dependencies': 'Check library dependencies'
}

def validate_security_checks(config: Dict[str, Any]) -> Set[str]:
    """Validate and get enabled security checks from config.
    
    Args:
        config: Security configuration dictionary
        
    Returns:
        Set of enabled security check names
        
    Raises:
        SecurityViolationError: If invalid security checks are specified
    """
    # Get security checks from config, default to all checks enabled
    security_checks = set(config.get('security_checks', AVAILABLE_SECURITY_CHECKS.keys()))
    
    # Validate each check
    invalid_checks = security_checks - set(AVAILABLE_SECURITY_CHECKS.keys())
    if invalid_checks:
        logger.error("Invalid security checks specified", extra={
            'payload': {
                'invalid_checks': list(invalid_checks),
                'available_checks': list(AVAILABLE_SECURITY_CHECKS.keys()),
                'component': 'security',
                'operation': 'validate_checks'
            }
        })
        raise SecurityViolationError(
            f"Invalid security checks specified: {', '.join(invalid_checks)}. "
            f"Available checks are: {', '.join(AVAILABLE_SECURITY_CHECKS.keys())}"
        )
    
    logger.debug("Security checks validated", extra={
        'payload': {
            'enabled_checks': list(security_checks),
            'component': 'security',
            'operation': 'validate_checks'
        }
    })
    
    return security_checks

# Initialize these as None and set them up only when needed
_rate_limiter = None
_subprocess_policy = None

def get_rate_limiter(config: Dict[str, Any]) -> Optional['SubprocessRateLimiter']:
    """Get or initialize the rate limiter based on configuration.
    
    Args:
        config: Security configuration dictionary
        
    Returns:
        Initialized rate limiter or None if disabled
    """
    global _rate_limiter
    
    if _rate_limiter is not None:
        return _rate_limiter
        
    rate_limit_config = config.get("subprocess_policy", {}).get("rate_limit", {})
    if rate_limit_config.get("enabled", False):
        _rate_limiter = SubprocessRateLimiter(
            rate_limit_config.get("max_calls", 10),
            rate_limit_config.get("period_seconds", 60)
        )
        logger.debug("Rate limiter initialized", extra={
            'payload': {
                **rate_limit_config,
                'component': 'security'
            }
        })
    else:
        logger.debug("Rate limiting disabled")
        _rate_limiter = None
        
    return _rate_limiter

def get_subprocess_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get or initialize the subprocess policy configuration.
    
    Args:
        config: Security configuration dictionary
        
    Returns:
        Subprocess policy configuration with the following keys:
        - level: Policy level ('none', 'system_tool', 'always')
        - allowed_commands: Set of allowed command patterns
        - blocked_commands: Set of blocked command patterns
    """
    policy = config.get("subprocess_policy", {})
    
    # Get policy level with proper default
    # 'none' is the most restrictive default for security
    policy_level = policy.get("level", "none")
    
    # Validate policy level
    valid_levels = {'none', 'system_tool', 'always'}
    if policy_level not in valid_levels:
        logger.warning(f"Invalid policy level '{policy_level}', defaulting to 'none'", extra={
            'payload': {
                'invalid_level': policy_level,
                'valid_levels': list(valid_levels),
                'component': 'security',
                'operation': 'get_policy'
            }
        })
        policy_level = "none"
    
    return {
        "level": policy_level,
        "allowed_commands": set(policy.get("allowed_commands", [])),
        "blocked_commands": set(policy.get("blocked_commands", []))
    }

def extract_command(node: ast.Call) -> Optional[str]:
    """Extract command from a subprocess call node.
    
    Handles various command formats:
    - String literals: subprocess.run("ls -l")
    - List literals: subprocess.run(["ls", "-l"])
    - f-strings: subprocess.run(f"ls {path}")
    - Join calls: subprocess.run(" ".join(["ls", "-l"]))
    
    Args:
        node: AST Call node representing the subprocess call
        
    Returns:
        Extracted command string or None if command can't be extracted
    """
    if not node.args:
        return None
        
    first_arg = node.args[0]
    
    # Handle string literal
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
        
    # Handle list literal
    elif isinstance(first_arg, ast.List):
        parts = []
        for elt in first_arg.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                # Quote arguments that contain spaces
                if ' ' in elt.value:
                    parts.append(f'"{elt.value}"')
                else:
                    parts.append(elt.value)
            else:
                logger.warning("Non-string element in command list", extra={
                    'payload': {
                        'element_type': type(elt).__name__,
                        'component': 'security',
                        'operation': 'command_extraction'
                    }
                })
                return None
        return ' '.join(parts)
        
    # Handle f-strings
    elif isinstance(first_arg, ast.JoinedStr):
        # F-strings are too dynamic to safely analyze
        logger.warning("F-string in subprocess command", extra={
            'payload': {
                'component': 'security',
                'operation': 'command_extraction',
                'warning': 'f-string_detected'
            }
        })
        return None
        
    # Handle string join
    elif (isinstance(first_arg, ast.Call) and 
          isinstance(first_arg.func, ast.Attribute) and 
          first_arg.func.attr == 'join' and 
          isinstance(first_arg.func.value, ast.Constant) and
          isinstance(first_arg.func.value.value, str)):
        if len(first_arg.args) == 1 and isinstance(first_arg.args[0], ast.List):
            parts = []
            for elt in first_arg.args[0].elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if ' ' in elt.value:
                        parts.append(f'"{elt.value}"')
                    else:
                        parts.append(elt.value)
                else:
                    logger.warning("Non-string element in join arguments", extra={
                        'payload': {
                            'element_type': type(elt).__name__,
                            'component': 'security',
                            'operation': 'command_extraction'
                        }
                    })
                    return None
            return ' '.join(parts)
            
    logger.warning("Unsupported command format", extra={
        'payload': {
            'arg_type': type(first_arg).__name__,
            'component': 'security',
            'operation': 'command_extraction'
        }
    })
    return None

def parse_command(command: str) -> List[str]:
    """Parse a command string into a list of arguments using shlex.
    
    Args:
        command: Command string to parse
        
    Returns:
        List of command arguments
        
    Raises:
        ValueError: If command parsing fails
    """
    try:
        return shlex.split(command)
    except ValueError as e:
        logger.error("Failed to parse command", extra={
            'payload': {
                'command': command,
                'error': str(e),
                'component': 'security',
                'operation': 'command_parsing'
            }
        })
        raise ValueError(f"Invalid command format: {str(e)}")

def validate_command(command: str) -> None:
    """Validate a command string against security rules.
    
    Args:
        command: The command string to validate
        
    Raises:
        SecurityViolationError: If the command violates security rules
    """
    try:
        # Parse command into arguments
        args = parse_command(command)
        if not args:
            raise SecurityViolationError("Empty command")
            
        # Get base command (first argument)
        base_cmd = args[0]
        
        # Check for shell metacharacters in the base command
        shell_metacharacters = ['|', '&', ';', '$', '>', '<', '`']
        if any(char in base_cmd for char in shell_metacharacters):
            logger.warning("Command contains shell metacharacters", extra={
                'payload': {
                    'command': command,
                    'base_command': base_cmd,
                    'metacharacters': [char for char in shell_metacharacters if char in base_cmd],
                    'component': 'security',
                    'operation': 'validate_command'
                }
            })
            raise SecurityViolationError(f"Command '{base_cmd}' contains disallowed shell metacharacters")
        
        # Check command length
        if len(command) > 1000:  # Arbitrary limit
            logger.warning("Command exceeds length limit", extra={
                'payload': {
                    'command_length': len(command),
                    'limit': 1000,
                    'component': 'security',
                    'operation': 'validate_command'
                }
            })
            raise SecurityViolationError("Command length exceeds security limit")
        
        # Check for suspicious patterns in the full command
        suspicious_patterns = ['rm -rf', 'mkfs', 'dd if=', 'wget', 'curl']
        for pattern in suspicious_patterns:
            if any(pattern in arg.lower() for arg in args):
                logger.warning("Command contains suspicious pattern", extra={
                    'payload': {
                        'command': command,
                        'pattern': pattern,
                        'component': 'security',
                        'operation': 'validate_command'
                    }
                })
                raise SecurityViolationError(f"Command contains suspicious pattern: {pattern}")
                
    except ValueError as e:
        raise SecurityViolationError(f"Command validation failed: {str(e)}")

class SubprocessRateLimiter:
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls: List[float] = []
    
    def check_rate_limit(self) -> bool:
        """Check if subprocess call is allowed under rate limit"""
        current_time = time.time()
        # Remove old calls outside the window
        self.calls = [t for t in self.calls if current_time - t <= self.period_seconds]
        
        if len(self.calls) >= self.max_calls:
            logger.warning("Subprocess rate limit exceeded", extra={
                'payload': {
                    'max_calls': self.max_calls,
                    'period_seconds': self.period_seconds,
                    'current_calls': len(self.calls),
                    'oldest_call': min(self.calls) if self.calls else None,
                    'time_until_next': (min(self.calls) + self.period_seconds - current_time) if self.calls else 0
                }
            })
            return False
            
        self.calls.append(current_time)
        logger.debug("Subprocess call allowed by rate limiter", extra={
            'payload': {
                'max_calls': self.max_calls,
                'period_seconds': self.period_seconds,
                'current_calls': len(self.calls),
                'calls_remaining': self.max_calls - len(self.calls)
            }
        })
        return True

class SecurityVisitor(ast.NodeVisitor):
    """AST visitor for security checks."""
    
    def __init__(self, config: Dict[str, Any], ephemeral_dir: Optional[str] = None, invoked_by_tool: bool = False):
        """Initialize security visitor.
        
        Args:
            config: Security configuration dictionary
            ephemeral_dir: Optional directory for temporary files
            invoked_by_tool: Whether this visitor is invoked by a system tool
        """
        super().__init__()
        self.config = config
        self.ephemeral_dir = ephemeral_dir
        self.invoked_by_tool = invoked_by_tool
        self.violations = {}
        self.enabled_checks = validate_security_checks(config)
        
        # Initialize library manager
        self.library_manager = LibraryManager(config)
        logger.debug("Library manager initialized")
        
        # Initialize rate limiter
        self.rate_limiter = get_rate_limiter(config)
        
        # Initialize subprocess policy
        self.subprocess_policy = get_subprocess_policy(config)
        
        logger.debug("Initialized SecurityVisitor")
    
    def _should_check(self, check_name: str) -> bool:
        """Check if a security check is enabled."""
        return check_name in self.enabled_checks
    
    def visit_Import(self, node: ast.Import):
        """Check import statements."""
        if not self._should_check('imports'):
            return
            
        for name in node.names:
            if name.name in self.config.get('blocked_modules', []):
                self._add_violation('imports', f"Import of module '{name.name}' is blocked")
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check import from statements."""
        if not self._should_check('imports'):
            return
            
        module = node.module or ''
        if module in self.config.get('blocked_modules', []):
            self._add_violation('imports', f"Import from module '{module}' is blocked")
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {'system', 'popen'}:
                self._add_violation('system_calls', f"Direct {func_name}() call detected")
        
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'subprocess':
                    command = extract_command(node)
                    if command:
                        self.check_subprocess_call(node, command)
                elif node.func.value.id == 'os' and node.func.attr in {'system', 'popen'}:
                    self._add_violation('system_calls', f"os.{node.func.attr}() call detected")
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function definitions."""
        if self._should_check('nested_functions'):
            # Check for nested function definitions
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child is not node:
                    self._add_violation('nested_functions', "Nested function definition detected")
                    break
        
        # Always traverse into the function body to check for other violations
        self.generic_visit(node)
    
    def visit_Global(self, node: ast.Global):
        """Check global statements."""
        if not self._should_check('globals'):
            return
            
        for name in node.names:
            self._add_violation('globals', f"Global statement for '{name}' detected")
        
        self.generic_visit(node)
    
    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Check nonlocal statements."""
        if not self._should_check('nonlocals'):
            return self.generic_visit(node)
            
        # Add violations for each nonlocal name
        for name in node.names:
            logger.warning(f"Nonlocal statement detected: {name}", extra={
                'payload': {
                    'name': name,
                    'component': 'security',
                    'operation': 'check_nonlocal'
                }
            })
            self._add_violation('nonlocals', "nonlocal statement detected")
        
        # Continue traversing child nodes
        self.generic_visit(node)
    
    def _add_violation(self, check: str, message: str):
        """Add a security violation."""
        if check not in self.violations:
            self.violations[check] = []
        self.violations[check].append(message)
    
    def finalize(self) -> Dict[str, List[str]]:
        """Finalize security checks and return violations.
        
        Returns:
            Dictionary mapping check names to lists of violation messages
        """
        # Just return the violations, let validate_code_security handle raising the error
        return self.violations
    
    def check_subprocess_call(self, node: ast.Call, command: str) -> None:
        """Check if a subprocess call is allowed by policy.
        
        Args:
            node: AST Call node representing the subprocess call
            command: Extracted command string
        """
        # Check for shell=True
        for kw in node.keywords:
            if kw.arg == 'shell' and isinstance(kw.value, ast.Constant) and kw.value.value:
                logger.warning("Subprocess call denied: shell execution not allowed", extra={
                    'payload': {
                        'command': command,
                        'component': 'security',
                        'operation': 'check_subprocess'
                    }
                })
                self._add_violation("subprocess", "shell execution")
                return
        
        # Parse command into parts
        try:
            command_parts = parse_command(command)
            if not command_parts:
                self._add_violation("subprocess", "Empty command")
                return
        except Exception as e:
            self._add_violation("subprocess", f"Invalid command format: {str(e)}")
            return
        
        # Get base command (first part)
        base_command = command_parts[0]
        
        # Check policy level
        if self.subprocess_policy["level"] == "none":
            logger.warning("Subprocess call denied: policy level none", extra={
                'payload': {
                    'command': command,
                    'component': 'security',
                    'operation': 'check_subprocess'
                }
            })
            self._add_violation("subprocess", "Subprocess calls are not allowed under default policy")
            return
        
        # Check blocked commands first (applies to all policy levels)
        if base_command in self.subprocess_policy.get("blocked_commands", []):
            logger.warning("Subprocess call denied: blocked command", extra={
                'payload': {
                    'command': command,
                    'base_command': base_command,
                    'component': 'security',
                    'operation': 'check_subprocess'
                }
            })
            self._add_violation("subprocess", "blocked command")
            return
        
        # Check allowed commands based on policy level
        if self.subprocess_policy["level"] == "system_tool":
            if not self.invoked_by_tool:
                logger.warning("Subprocess call denied: system tool policy", extra={
                    'payload': {
                        'command': command,
                        'base_command': base_command,
                        'component': 'security',
                        'operation': 'check_subprocess'
                    }
                })
                self._add_violation("subprocess", "system tool")
                return
            
            if base_command not in self.subprocess_policy.get("allowed_commands", []):
                logger.warning("Subprocess call denied: not an allowed system tool", extra={
                    'payload': {
                        'command': command,
                        'base_command': base_command,
                        'component': 'security',
                        'operation': 'check_subprocess'
                    }
                })
                self._add_violation("subprocess", "system tool")
                return
                
        elif self.subprocess_policy["level"] == "always":
            allowed_commands = self.subprocess_policy.get("allowed_commands", [])
            if allowed_commands and base_command not in allowed_commands:
                logger.warning("Subprocess call denied: not in allowed list", extra={
                    'payload': {
                        'command': command,
                        'base_command': base_command,
                        'component': 'security',
                        'operation': 'check_subprocess'
                    }
                })
                self._add_violation("subprocess", f"Command '{base_command}' is not in allowed commands list")
                return
        
        # Check rate limit if enabled
        if self.rate_limiter and not self.rate_limiter.check_rate_limit():
            logger.warning("Subprocess call denied: rate limit exceeded", extra={
                'payload': {
                    'command': command,
                    'component': 'security',
                    'operation': 'check_subprocess'
                }
            })
            self._add_violation("subprocess", "rate limit")
            return
        
        # If we get here, the command is allowed
        logger.debug(f"Subprocess call allowed: {command}", extra={
            'payload': {
                'command': command,
                'policy_level': self.subprocess_policy["level"],
                'component': 'security',
                'operation': 'check_subprocess'
            }
        })
    
    def visit(self, node: ast.AST) -> None:
        """Override visit to ensure violations are properly propagated."""
        # First visit the node and its children
        super().visit(node)
        
        # After visiting, check if we have any violations
        if self.violations:
            # Format violation messages
            violation_messages = []
            for check, messages in self.violations.items():
                for message in messages:
                    violation_messages.append(f"{check}: {message}")
            
            logger.warning("Security violations found", extra={
                'payload': {
                    'violations': self.violations,
                    'component': 'security',
                    'operation': 'validate'
                }
            })
            raise SecurityViolationError(
                f"Security violations found: {', '.join(violation_messages)}",
                str(node),
                {'violations': self.violations}
            )

def validate_code_security(code: str, config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate code against security configuration.
    
    Args:
        code: Python code to validate
        config: Security configuration dictionary
        
    Returns:
        Dictionary mapping check names to lists of violation messages
        
    Raises:
        SecurityViolationError: If any security violations are found
    """
    logger.debug("Starting security validation")
    
    # Validate security checks
    enabled_checks = validate_security_checks(config)
    
    # Initialize library manager
    library_manager = LibraryManager(config)
    logger.debug("Library manager initialized")
    
    # Initialize rate limiter
    rate_limiter = get_rate_limiter(config)
    
    # Initialize subprocess policy
    subprocess_policy = get_subprocess_policy(config)
    
    # Parse code into AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityViolationError(
            f"Syntax error in code: {str(e)}",
            code,
            {'error': str(e)}
        )
    
    # Visit AST with security visitor
    visitor = SecurityVisitor(config)
    visitor.visit(tree)
    
    # Get violations
    violations = visitor.finalize()
    
    # Check for violations
    if violations:
        # Format violation messages
        violation_messages = []
        for check, messages in violations.items():
            for message in messages:
                violation_messages.append(f"{check}: {message}")
            
        logger.warning("Security violations found", extra={
            'payload': {
                'violations': violations,
                'component': 'security',
                'operation': 'validate'
            }
        })
        raise SecurityViolationError(
            f"Security violations found: {', '.join(violation_messages)}",
            code,
            {'violations': violations}
        )
    
    logger.debug("Security validation passed")
    return violations

def create_safe_execution_environment() -> Dict[str, Any]:
    """
    Create a restricted environment for code execution
    
    Returns:
        Dictionary containing allowed builtins and modules
    """
    import builtins
    
    # Start with minimal builtins
    safe_builtins = {
        name: getattr(builtins, name)
        for name in [
            'len', 'range', 'enumerate',
            'str', 'int', 'float', 'bool',
            'list', 'dict', 'set', 'tuple',
            'sum', 'min', 'max',
            'sorted', 'reversed',
            'zip', 'map', 'filter',
            'any', 'all',
            'abs', 'round', 'pow', 'divmod',
            'isinstance', 'issubclass',
            'hasattr', 'getattr', 'setattr', 'delattr',
            'property', 'staticmethod', 'classmethod'
        ]
    }
    
    # Add safe modules
    safe_modules = {
        'math': __import__('math'),
        'random': __import__('random'),
        'datetime': __import__('datetime'),
        'decimal': __import__('decimal'),
        'collections': __import__('collections'),
        'itertools': __import__('itertools'),
        'functools': __import__('functools'),
        'operator': __import__('operator'),
        're': __import__('re')
    }
    
    return {
        '__builtins__': safe_builtins,
        **safe_modules
    } 

if __name__ == "__main__":
    # Example usage and tests
    config = {
        "subprocess_policy": {
            "level": "prompt",
            "allowed_commands": ["ls", "git"],
            "blocked_commands": ["rm", "mkfs"],
            "rate_limit": {
                "enabled": True,
                "max_calls": 10,
                "period_seconds": 60
            }
        }
    }
    
    # Initialize subprocess policy and rate limiter
    policy = get_subprocess_policy(config)
    limiter = get_rate_limiter(config)
    
    # Example validation
    try:
        validate_command("ls -l")
        print("Command validation passed")
    except SecurityViolationError as e:
        print(f"Command validation failed: {e}") 