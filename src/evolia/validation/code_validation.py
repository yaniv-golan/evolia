"""Code validation and execution utilities"""
import ast
import logging
import inspect
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

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

        jsonschema.validate(instance=data, schema=schema)
        return data
    except jsonschema.exceptions.ValidationError as e:
        raise ValidationError(str(e), validation_results={"schema_errors": [str(e)]})
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
            - inputs: List of input arguments
            - expected: Expected output
        timeout: Maximum execution time per test case

    Returns:
        Dictionary containing test results
    """
    logger.debug(
        "Executing test cases",
        extra={"payload": {"test_count": len(test_cases), "timeout": timeout}},
    )

    results = {"passed": 0, "failed": 0, "failures": [], "error": None}

    # Create isolated namespace
    namespace = {}

    try:
        # Execute code to define function
        exec(code, namespace)

        # Find the function
        func_name = None
        for name, obj in namespace.items():
            if inspect.isfunction(obj):
                func_name = name
                break

        if not func_name:
            raise ValueError("No function found in code")

        func = namespace[func_name]

        # Run test cases
        for i, test in enumerate(test_cases):
            try:
                args = test.get("inputs", [])
                expected = test.get("expected")

                # Execute with timeout
                import signal

                def handler(signum, frame):
                    raise TimeoutError("Timeout")

                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)

                try:
                    actual = func(*args)
                    signal.alarm(0)  # Disable alarm

                    if actual == expected:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                        results["failures"].append(
                            {
                                "test_case": i,
                                "inputs": args,
                                "expected": expected,
                                "actual": actual,
                            }
                        )
                except TimeoutError:
                    results["failed"] += 1
                    results["failures"].append(
                        {"test_case": i, "inputs": args, "error": "Timeout"}
                    )
                finally:
                    signal.alarm(0)  # Ensure alarm is disabled

            except Exception as e:
                results["failed"] += 1
                results["failures"].append(
                    {"test_case": i, "inputs": test.get("inputs", []), "error": str(e)}
                )

    except Exception as e:
        logger.error(f"Error executing tests: {str(e)}", exc_info=True)
        results["error"] = str(e)

    logger.debug("Test execution complete", extra={"payload": {"results": results}})

    return results
