"""Integration tests for code execution functionality."""
import ast
import math

import pytest

from evolia.core.restricted_execution import RestrictedExecutor, restricted_import


def execute_test_cases(code: str, test_cases: list, timeout: int = 5) -> dict:
    """Execute test cases for a given code snippet.

    Args:
        code: The Python code to test
        test_cases: List of test cases with inputs and expected outputs
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with test results including passed/failed counts and failure details
    """
    executor = RestrictedExecutor(
        allowed_modules={"math", "typing"},
        allowed_builtins={"len", "str", "int", "float", "sum"},
    )

    results = {"passed": 0, "failed": 0, "failures": []}

    # Parse AST to get function info
    tree = ast.parse(code)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not functions:
        raise ValueError("No function found in code")
    func = functions[0]  # Take first function
    func_name = func.name

    # Get parameter names from function definition
    param_names = [arg.arg for arg in func.args.args]

    for case in test_cases:
        try:
            # Set up timeout using signal
            import signal

            def handler(signum, frame):
                raise TimeoutError("Execution timed out")

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

            try:
                # Convert inputs to named arguments using actual parameter names
                inputs = case["inputs"]
                if isinstance(inputs, list):
                    if len(inputs) != len(param_names):
                        raise ValueError(
                            f"Expected {len(param_names)} arguments, got {len(inputs)}"
                        )
                    inputs = dict(zip(param_names, inputs))

                result = executor.execute_in_sandbox(code, inputs, ".")
                signal.alarm(0)  # Disable alarm

                if result == case["expected"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append(
                        {
                            "inputs": case["inputs"],
                            "expected": case["expected"],
                            "actual": result,
                            "error": "Output mismatch",
                        }
                    )
            except TimeoutError:
                results["failed"] += 1
                results["failures"].append(
                    {
                        "inputs": case["inputs"],
                        "expected": case["expected"],
                        "error": "Execution timed out",
                    }
                )
            except Exception as e:
                results["failed"] += 1
                results["failures"].append(
                    {
                        "inputs": case["inputs"],
                        "expected": case["expected"],
                        "error": str(e),
                    }
                )
            finally:
                signal.alarm(0)  # Ensure alarm is disabled

        except Exception as e:
            results["failed"] += 1
            results["failures"].append(
                {
                    "inputs": case["inputs"],
                    "expected": case["expected"],
                    "error": str(e),
                }
            )

    return results


def test_execute_basic_test_cases():
    """Test execution of basic test cases."""
    executor = RestrictedExecutor(
        allowed_modules={"math", "typing"},
        allowed_builtins={"len", "str", "int", "float"},
    )

    code = """
def add(a, b):
    return a + b
"""
    result = executor.execute_in_sandbox(code, {"a": 1, "b": 2}, ".")
    assert result == 3


def test_execute_failing_test_cases():
    """Test execution of failing test cases."""
    code = """
def subtract(a, b):
    return a + b  # Wrong operation
"""
    test_cases = [{"inputs": [3, 2], "expected": 1}, {"inputs": [1, 1], "expected": 0}]

    results = execute_test_cases(code, test_cases)
    assert results["passed"] == 0
    assert results["failed"] == 2
    assert len(results["failures"]) == 2


def test_execute_timeout():
    """Test execution with timeout."""
    code = """
def infinite_loop():
    x = 0
    while True:
        x = x + 1
    return x
"""
    test_cases = [{"inputs": [], "expected": None}]

    results = execute_test_cases(code, test_cases, timeout=1)
    assert results["failed"] == 1
    error_msg = results["failures"][0]["error"].lower()
    assert any(phrase in error_msg for phrase in ["timeout", "timed out"])


def test_execute_runtime_error():
    """Test execution with runtime error."""
    code = """
def divide(a, b):
    return a / b
"""
    test_cases = [{"inputs": [1, 0], "expected": None}]

    results = execute_test_cases(code, test_cases)
    assert results["failed"] == 1
    assert "error" in results["failures"][0]


def test_execute_with_imports():
    """Test execution with allowed imports."""
    code = """
import math
def calculate_circle_area(radius: float) -> float:
    return math.pi * radius ** 2
"""
    test_cases = [
        {"inputs": [1], "expected": math.pi},
        {"inputs": [2], "expected": 4 * math.pi},
    ]

    print("\nRunning test cases:")  # Debug print
    results = execute_test_cases(code, test_cases)
    print(f"\nTest results: {results}")  # Debug print

    if results["failed"] > 0:
        print("\nFailures:")  # Debug print
        for failure in results["failures"]:
            print(f"  {failure}")  # Debug print

    assert results["passed"] == 2
    assert results["failed"] == 0


def test_execute_with_type_validation():
    """Test execution with type validation."""
    code = """
def process_list(items: list) -> int:
    return sum(items)
"""
    test_cases = [
        {"inputs": [[1, 2, 3]], "expected": 6},
        {"inputs": [["not", "integers"]], "expected": None},  # Should fail type check
    ]

    results = execute_test_cases(code, test_cases)
    assert results["passed"] == 1
    assert results["failed"] == 1
    assert any("type" in str(error).lower() for error in results["failures"])
