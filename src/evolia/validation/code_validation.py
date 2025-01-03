"""Code validation and execution utilities"""
import ast
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("evolia")


@dataclass
class ValidationResult:
    """Result of code validation"""

    is_valid: bool
    issues: List[str]
    details: Dict[str, Any]

    def get_error_messages(self) -> List[str]:
        """Return the list of validation issues"""
        return self.issues


def validate_schema(data: dict, schema_type: Any) -> Dict[str, Any]:
    """Validate data against a schema type.

    Args:
        data: The data to validate
        schema_type: The schema type to validate against

    Returns:
        Dict[str, Any]: The validated data

    Raises:
        ValidationError: If validation fails
    """
    from evolia.utils.exceptions import ValidationError

    try:
        # Convert schema type to dict if needed
        if hasattr(schema_type, "schema"):
            schema = schema_type.schema()
        else:
            schema = schema_type

        import jsonschema
        from jsonschema.validators import validator_for

        # Create a validator that collects all errors
        validator = validator_for(schema)(schema)
        errors = list(validator.iter_errors(data))

        if errors:
            # Combine all error messages
            error_messages = []
            for error in errors:
                path = " -> ".join(str(p) for p in error.path) if error.path else "root"
                error_messages.append(f"{path}: {error.message}")

            raise ValidationError(
                "Schema validation failed:\n" + "\n".join(error_messages),
                details={"schema_errors": error_messages},
            )

        return data
    except jsonschema.exceptions.ValidationError as e:
        raise ValidationError(str(e), details={"schema_errors": [str(e)]})
    except Exception as e:
        raise ValidationError(f"Schema validation error: {str(e)}")


def _has_nested_functions(node: ast.AST) -> bool:
    """Check if an AST node contains nested function definitions"""
    # If this is a function definition, look for other function definitions in its body
    if isinstance(node, ast.FunctionDef):
        return any(
            isinstance(child, ast.FunctionDef)
            for child in ast.walk(node)
            if child is not node
        )
    return False


def _check_undefined_type_hints(tree: ast.AST) -> List[str]:
    """Check for undefined type hints in the AST.

    Args:
        tree: AST to check

    Returns:
        List of error messages for undefined type hints
    """
    issues = []
    defined_names = set()

    # First pass: collect all defined names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                defined_names.add(name.name)
                if name.asname:
                    defined_names.add(name.asname)
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                defined_names.add(name.name)
                if name.asname:
                    defined_names.add(name.asname)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)

    # Second pass: check annotations for undefined names
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name):
            if (
                node.annotation.id not in defined_names
                and node.annotation.id not in __builtins__
            ):
                issues.append(f"Undefined type hint: {node.annotation.id}")
        elif isinstance(node, ast.FunctionDef):
            # Check return annotation
            if node.returns and isinstance(node.returns, ast.Name):
                if (
                    node.returns.id not in defined_names
                    and node.returns.id not in __builtins__
                ):
                    issues.append(f"Undefined return type hint: {node.returns.id}")
            elif node.returns and isinstance(node.returns, ast.Subscript):
                if isinstance(node.returns.value, ast.Name):
                    if (
                        node.returns.value.id not in defined_names
                        and node.returns.value.id not in __builtins__
                    ):
                        issues.append(f"Undefined type hint: {node.returns.value.id}")

            # Check argument annotations
            for arg in node.args.args:
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        if (
                            arg.annotation.id not in defined_names
                            and arg.annotation.id not in __builtins__
                        ):
                            issues.append(
                                f"Undefined parameter type hint: {arg.annotation.id}"
                            )
                    elif isinstance(arg.annotation, ast.Subscript):
                        if isinstance(arg.annotation.value, ast.Name):
                            if (
                                arg.annotation.value.id not in defined_names
                                and arg.annotation.value.id not in __builtins__
                            ):
                                issues.append(
                                    f"Undefined type hint: {arg.annotation.value.id}"
                                )

    return issues


def _check_security_constraints(tree: ast.AST, constraints: List[str]) -> List[str]:
    """Check for security violations in the AST.

    Args:
        tree: AST to check
        constraints: List of security constraints to enforce

    Returns:
        List of security violation messages
    """
    issues = []

    for node in ast.walk(tree):
        # Check for system calls (os.system, subprocess.run, etc)
        if "no_system_calls" in constraints:
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "system"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os"
                    ):
                        issues.append("Security violation: os.system call detected")
                    elif (
                        node.func.attr == "run"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "subprocess"
                    ):
                        issues.append(
                            "Security violation: subprocess.run call detected"
                        )
                    elif (
                        node.func.attr == "popen"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os"
                    ):
                        issues.append("Security violation: os.popen call detected")

        # Check for code injection functions (eval, exec, etc)
        if "no_eval" in constraints:
            if isinstance(node, ast.Call):
                dangerous_functions = {
                    "eval": "code evaluation",
                    "exec": "code execution",
                    "compile": "code compilation",
                    "__import__": "dynamic imports",
                }

                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        issues.append(
                            f"Security violation: Dangerous {dangerous_functions[node.func.id]} using {node.func.id}()"
                        )
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in dangerous_functions:
                        issues.append(
                            f"Security violation: Dangerous {dangerous_functions[node.func.attr]} using {node.func.attr}()"
                        )

        # Check for dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dangerous_modules = {
                "subprocess": "subprocess operations",
                "socket": "network operations",
                "pickle": "unsafe deserialization",
                "marshal": "unsafe deserialization",
            }

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_modules:
                        issues.append(
                            f"Security violation: Dangerous import of {alias.name} for {dangerous_modules[alias.name]}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module in dangerous_modules:
                    issues.append(
                        f"Security violation: Dangerous import from {node.module} for {dangerous_modules[node.module]}"
                    )

    return issues


def _check_file_operations(node: ast.AST, allowed_paths: List[str]) -> List[str]:
    """Check file operations for path violations.

    Args:
        node: AST node to check
        allowed_paths: List of allowed file paths

    Returns:
        List of file access violation messages
    """
    issues = []

    if isinstance(node, ast.Call):
        # Check open() calls
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            if len(node.args) >= 1:
                # Get the file path argument
                path_arg = node.args[0]
                if isinstance(path_arg, ast.Constant):
                    file_path = path_arg.value
                    # Check if path is allowed
                    if not any(
                        str(file_path).startswith(allowed) for allowed in allowed_paths
                    ):
                        issues.append(
                            f"File access violation: Attempted to access {file_path} outside allowed paths {allowed_paths}"
                        )

        # Check file operations through pathlib
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "Path":
                if node.func.attr in {"open", "write_text", "write_bytes", "touch"}:
                    # Get the file path from Path construction
                    if len(node.args) >= 1:
                        path_arg = node.args[0]
                        if isinstance(path_arg, ast.Constant):
                            file_path = path_arg.value
                            if not any(
                                str(file_path).startswith(allowed)
                                for allowed in allowed_paths
                            ):
                                issues.append(
                                    f"File access violation: Attempted to access {file_path} outside allowed paths {allowed_paths}"
                                )

    return issues


def _check_allowed_modules(tree: ast.AST, allowed_modules: Set[str]) -> List[str]:
    """Check that only allowed modules are imported.

    Args:
        tree: AST to check
        allowed_modules: Set of module names that are allowed to be imported

    Returns:
        List of module violation messages
    """
    issues = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]  # Get base module name
                if module_name not in allowed_modules:
                    issues.append(f"Module not allowed: {module_name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]  # Get base module name
                if module_name not in allowed_modules:
                    issues.append(f"Module not allowed: {module_name}")

    return issues


def validate_python_code(code: str, requirements: Dict[str, Any]) -> ValidationResult:
    """
    Validate Python code against requirements

    Args:
        code: The Python code to validate
        requirements: Dictionary containing:
            - function_name: Expected function name
            - parameters: List of parameter names or Parameter objects
            - return_type: Expected return type annotation
            - constraints: List of constraints to check
            - allowed_write_paths: List of paths where file writes are allowed
            - allowed_modules: Set of module names that are allowed to be imported

    Returns:
        ValidationResult containing validation status and details
    """
    logger.debug(
        "Validating Python code",
        extra={"payload": {"code_length": len(code), "requirements": requirements}},
    )

    issues = []
    details = {}

    # Parse AST
    try:
        tree = ast.parse(code)
        details["syntax_valid"] = True
    except SyntaxError as e:
        logger.error(f"Syntax error in code: {str(e)}")
        return ValidationResult(
            is_valid=False,
            issues=[f"Syntax error: {str(e)}"],
            details={"syntax_valid": False},
        )

    # Check security constraints
    if "constraints" in requirements:
        security_issues = _check_security_constraints(tree, requirements["constraints"])
        if security_issues:
            issues.extend(security_issues)

    # Check file operations if allowed paths are specified
    if "allowed_write_paths" in requirements:
        for node in ast.walk(tree):
            file_issues = _check_file_operations(
                node, requirements["allowed_write_paths"]
            )
            if file_issues:
                issues.extend(file_issues)

    # Check allowed modules
    if "allowed_modules" in requirements:
        module_issues = _check_allowed_modules(tree, requirements["allowed_modules"])
        if module_issues:
            issues.extend(module_issues)

    # Find function definition
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    if not functions:
        logger.error("No function definition found")
        return ValidationResult(
            is_valid=False,
            issues=["No function definition found"],
            details={"has_function": False},
        )

    func = functions[0]  # Take first function for now
    details["function_name"] = func.name

    # Check function name
    if "function_name" in requirements:
        expected_name = requirements["function_name"]
        if func.name != expected_name:
            issues.append(
                f"Function name mismatch: expected '{expected_name}', got '{func.name}'"
            )

    # Check parameters
    if "parameters" in requirements:
        expected_params = requirements["parameters"]
        actual_params = [arg.arg for arg in func.args.args]
        details["parameters"] = actual_params

        if len(actual_params) != len(expected_params):
            issues.append(
                f"Parameter count mismatch: expected {len(expected_params)}, got {len(actual_params)}"
            )
        else:
            for expected, actual in zip(expected_params, actual_params):
                # Extract parameter name based on type
                if isinstance(expected, dict):
                    expected_name = expected.get("name", "")
                elif hasattr(expected, "name"):  # Parameter object
                    expected_name = expected.name
                else:
                    # Handle string format: "name='param' type='type'"
                    try:
                        if isinstance(expected, str):
                            # Extract name value from format: name='value'
                            name_part = (
                                expected.split("name=")[1].split()[0].strip("'\"")
                            )
                            expected_name = name_part
                        else:
                            expected_name = str(expected)
                    except (IndexError, AttributeError):
                        expected_name = str(expected)

                if expected_name != actual:
                    issues.append(
                        f"Parameter name mismatch: expected parameter named '{expected_name}', got '{actual}'"
                    )

    # Check return type annotation if present
    if "return_type" in requirements and func.returns:
        try:
            return_type = ast.unparse(func.returns)
            details["return_type"] = return_type
            expected_type = requirements["return_type"]

            if return_type != expected_type:
                issues.append(
                    f"Return type mismatch: expected '{expected_type}', got '{return_type}'"
                )
        except Exception as e:
            logger.warning(f"Error checking return type: {str(e)}", exc_info=True)

    # Check constraints
    if "constraints" in requirements:
        for constraint in requirements["constraints"]:
            if constraint == "no_globals":
                globals_used = any(
                    isinstance(node, ast.Global) for node in ast.walk(tree)
                )
                if globals_used:
                    issues.append("Code uses global variables")

            elif constraint == "no_nested_functions":
                # Check each function definition for nested functions
                for func_def in functions:
                    if _has_nested_functions(func_def):
                        issues.append("Code contains nested function definitions")
                        break

    # Check for undefined type hints
    type_hint_issues = _check_undefined_type_hints(tree)
    if type_hint_issues:
        issues.extend(type_hint_issues)

    logger.debug(
        "Validation complete", extra={"payload": {"issues": issues, "details": details}}
    )

    return ValidationResult(is_valid=len(issues) == 0, issues=issues, details=details)


def execute_test_cases(
    code: str, test_cases: List[Dict[str, Any]], timeout: int = 5
) -> Dict[str, Any]:
    """
    Execute test cases against Python code

    Args:
        code: The Python code to test
        test_cases: List of test cases, each containing:
            - inputs: List of input arguments or dict of keyword arguments
            - expected: Expected output
        timeout: Maximum execution time per test case

    Returns:
        Dictionary containing test results
    """
    results = {"passed": 0, "failed": 0, "failures": [], "error": None}

    try:
        # Create restricted executor with common modules
        from evolia.core.restricted_execution import RestrictedExecutor

        executor = RestrictedExecutor(
            allowed_modules={"math", "typing", "datetime", "json", "re"},
            allowed_builtins={
                "len",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "tuple",
                "pow",
                "print",
            },
        )

        # Find the function name by parsing the AST
        tree = ast.parse(code)
        functions = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        if not functions:
            raise ValueError("No function found in code")
        func_name = functions[0]

        def values_match(actual: Any, expected: Any, epsilon: float = 1e-10) -> bool:
            """Compare values with special handling for floating point numbers."""
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(actual - expected) < epsilon
            return actual == expected

        # Run test cases
        for i, test in enumerate(test_cases):
            try:
                inputs = test.get("inputs", [])
                expected = test.get("expected")

                # Execute with timeout
                import signal

                def handler(signum, frame):
                    raise TimeoutError("Timeout")

                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)

                try:
                    # Execute in sandbox with the specific function name
                    # Convert inputs to a dictionary if it's a list
                    if isinstance(inputs, list):
                        # Create a dictionary with positional arguments
                        input_dict = {"inputs": inputs}
                    else:
                        input_dict = inputs

                    actual = executor.execute_in_sandbox(
                        script=code,
                        inputs=input_dict,
                        output_dir=".",
                        function_name=func_name,
                    )
                    signal.alarm(0)  # Disable alarm

                    if values_match(actual, expected):
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["failures"].append(
                            {
                                "test_case": i,
                                "inputs": inputs,
                                "expected": expected,
                                "actual": actual,
                            }
                        )
                except TimeoutError:
                    results["failed"] += 1
                    results["failures"].append(
                        {"test_case": i, "inputs": inputs, "error": "Timeout"}
                    )
                except Exception as e:
                    results["failed"] += 1
                    results["failures"].append(
                        {"test_case": i, "inputs": inputs, "error": str(e)}
                    )
                finally:
                    signal.alarm(0)  # Ensure alarm is disabled

            except Exception as e:
                results["failed"] += 1
                results["failures"].append(
                    {"test_case": i, "inputs": test.get("inputs", []), "error": str(e)}
                )

    except Exception as e:
        results["error"] = str(e)

    return results
