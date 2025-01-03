"""Restricted execution module for Evolia."""

import builtins
import copy
import importlib
import logging
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)

from ..utils.exceptions import EvoliaError, SecurityViolationError

logger = logging.getLogger(__name__)

REQUIRED_GUARDS = {
    "_getattr_",
    "_write_",
    "_getiter_",
    "_unpack_sequence_",
    "_getitem_",
    "_print_",
    "_inplacevar_",
    "__import__",
    "__builtins__",
}


def guarded_getitem(ob, index):
    """Restricted Python guard for item access"""
    if isinstance(ob, (dict, list, tuple, str)):
        return ob[index]
    # Allow DataFrame and Series access for pandas operations
    if "pandas" in sys.modules and isinstance(
        ob, (sys.modules["pandas"].DataFrame, sys.modules["pandas"].Series)
    ):
        return ob[index]
    raise TypeError(f"Restricted access to {type(ob).__name__} objects")


def guarded_write(ob):
    """Restricted Python guard for write operations"""
    if isinstance(ob, (str, int, float, bool, list, dict, tuple)):
        return ob
    raise SecurityViolationError(f"Writing {type(ob).__name__} objects is not allowed")


class RestrictedExecutionError(EvoliaError):
    """Raised when code violates RestrictedPython's security rules"""

    def __init__(self, message: str, code: str, details: Dict[str, Any]):
        super().__init__(message)
        self.code = code
        self.details = details


class RestrictedImportError(RestrictedExecutionError):
    """Raised when code attempts to import forbidden modules"""

    pass


class RestrictedAttributeError(RestrictedExecutionError):
    """Raised when code attempts to access forbidden attributes"""

    pass


def secure_open(filename, mode="r", *args, **kwargs):
    """Secure file open function that restricts access to output_dir."""
    if not isinstance(filename, (str, Path)):
        raise SecurityViolationError("Filename must be a string or Path object")

    path = Path(filename)

    # Get output_dir from the caller's globals
    frame = sys._getframe(1)
    while frame is not None:
        if "output_dir" in frame.f_globals:
            output_dir = frame.f_globals["output_dir"]
            break
        frame = frame.f_back
    if frame is None or not output_dir:
        raise SecurityViolationError("output_dir not found in globals")

    # Convert both paths to absolute and resolve any symlinks
    try:
        output_dir = Path(output_dir).resolve()
        # If path is relative, join with output_dir
        if not path.is_absolute():
            path = output_dir / path
        path = path.resolve()

        # Ensure the path is within output_dir
        if not str(path).startswith(str(output_dir)):
            raise SecurityViolationError(
                f"Access to {path} outside output_dir is not allowed"
            )
    except (OSError, RuntimeError) as e:
        raise SecurityViolationError(f"Error resolving path: {e}")

    # Only allow read/write operations
    if mode not in ("r", "w", "a", "r+", "w+", "a+"):
        raise SecurityViolationError(f"File mode {mode} is not allowed")

    return open(path, mode, *args, **kwargs)


def restricted_import(
    name, globals=None, locals=None, fromlist=(), level=0, allowed_modules=None
):
    """Restricted import function that only allows specified modules."""
    logger.debug(f"restricted_import called for module: {name}")
    if allowed_modules is None:
        raise SecurityViolationError("No modules are allowed to be imported")

    # Special handling for os and os.path - use the pre-created modules
    if name == "os.path":
        logger.debug("Handling os.path import")
        # For os.path import, return the path module from os
        frame = sys._getframe(1)
        while frame is not None:
            if "os" in frame.f_globals:
                return frame.f_globals["os"].path
            frame = frame.f_back
        raise ImportError("Could not find pre-created os module")
    elif name == "os":
        logger.debug("Handling os import")
        frame = sys._getframe(1)
        while frame is not None:
            if "os" in frame.f_globals:
                return frame.f_globals["os"]
            frame = frame.f_back
        raise ImportError("Could not find pre-created os module")

    # For all other modules, check if they're allowed
    logger.debug(
        f"Checking if module {name} is allowed. Allowed modules: {allowed_modules}"
    )
    if name not in allowed_modules:
        logger.debug(f"Module {name} is not allowed")
        raise RestrictedImportError(
            f"Import of module '{name}' is not allowed",
            name,
            {"allowed_modules": list(allowed_modules)},
        )

    try:
        # Perform normal import
        logger.debug(f"Attempting to import module {name}")
        module = importlib.import_module(name)

        # Add security check for module path
        if hasattr(module, "__file__"):
            module_path = Path(module.__file__)
            if not any(str(module_path).startswith(str(p)) for p in sys.path):
                raise SecurityViolationError(f"Module {name} is outside system path")

        # If this is a submodule import (e.g., os.path), return the submodule
        if "." in name:
            for part in name.split(".")[1:]:
                module = getattr(module, part)

        return module

    except ImportError as e:
        logger.debug(f"Import error for module {name}: {e}")
        raise RestrictedImportError(
            f"Failed to import allowed module '{name}': {str(e)}",
            name,
            {"error": str(e)},
        )


class RestrictedExecutor:
    """Handles execution of code in a restricted environment."""

    def __init__(self, allowed_modules: Set[str], allowed_builtins: Set[str]):
        """Initialize with allowed modules and builtins."""
        self.allowed_modules = allowed_modules
        self.allowed_builtins = allowed_builtins
        self.logger = logging.getLogger(__name__)

    def _validate_globals(self, globals_dict: Dict[str, Any]) -> None:
        """Validate globals dictionary for security.

        Args:
            globals_dict: Dictionary of globals to validate

        Raises:
            SecurityViolationError: If unauthorized guards or unsafe values are found
        """
        # Check for required guards
        missing_guards = REQUIRED_GUARDS - set(globals_dict.keys())
        if missing_guards:
            raise SecurityViolationError(f"Missing required guards: {missing_guards}")

        for key, value in globals_dict.items():
            if key.startswith("_") and key not in REQUIRED_GUARDS:
                raise SecurityViolationError(f"Unauthorized guard: {key}")
            if callable(value) and not key.startswith("_"):
                if key not in self.allowed_builtins:
                    raise SecurityViolationError(f"Unauthorized callable: {key}")

    def prepare_restricted_globals(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        allowed_modules: Optional[Set[str]] = None,
        existing_globals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare restricted globals for code execution."""
        logger.info("Preparing restricted globals")
        logger.debug(f"Allowed modules: {allowed_modules}")
        logger.debug(f"Allowed builtins: {self.allowed_builtins}")

        if allowed_modules is None:
            allowed_modules = self.allowed_modules

        # Start with existing globals or empty dict
        restricted_globals = copy.deepcopy(existing_globals) if existing_globals else {}

        # Validate existing globals if provided
        if existing_globals:
            self._validate_globals(existing_globals)

        # Create a restricted __builtins__ dictionary
        restricted_builtins = {}

        # Add safe builtins filtered by allowed_builtins
        restricted_builtins.update(
            {
                name: func
                for name, func in safe_builtins.items()
                if name in self.allowed_builtins
            }
        )

        # Add required guard functions
        guarded_builtins = {
            "_getattr_": safer_getattr,
            "_write_": guarded_write,  # Use our restricted write guard
            "_getiter_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_getitem_": guarded_getitem,
            "_print_": print,
            "_inplacevar_": lambda op, x, y: op(x, y),
        }

        # Add our restricted import function
        restricted_import_func = lambda name, globals=None, locals=None, fromlist=(), level=0: restricted_import(
            name,
            globals,
            locals,
            fromlist,
            level,
            allowed_modules=allowed_modules,
        )
        restricted_builtins["__import__"] = restricted_import_func
        restricted_globals["__import__"] = restricted_import_func
        restricted_globals["__builtins__"] = restricted_builtins

        # Add all required guards (they must all be present)
        for guard_name in REQUIRED_GUARDS:
            if guard_name not in guarded_builtins and guard_name not in (
                "__import__",
                "__builtins__",
            ):
                raise SecurityViolationError(f"Missing required guard: {guard_name}")
            if guard_name in guarded_builtins:
                restricted_globals[guard_name] = guarded_builtins[guard_name]

        # Add allowed Python builtins
        python_builtins = {
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "list": list,
            "tuple": tuple,
            "set": set,
            "dict": dict,
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "abs": abs,
            "all": all,
            "any": any,
            "min": min,
            "max": max,
            "round": round,
            "pow": pow,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "delattr": delattr,
            "callable": callable,
            "type": type,
            "open": secure_open,  # Add our secure_open as the open function
        }
        restricted_builtins.update(
            {
                name: func
                for name, func in python_builtins.items()
                if name in self.allowed_builtins
                or name == "open"  # Always allow secure_open
            }
        )

        # Create restricted os module
        restricted_os = types.ModuleType("os")
        restricted_os.path = types.ModuleType("os.path")

        # Add allowed os functions
        for func_name in ["listdir", "makedirs", "path", "remove", "unlink", "rmdir"]:
            if hasattr(os, func_name):
                setattr(restricted_os, func_name, getattr(os, func_name))

        # Add allowed os.path functions
        for func_name in ["join", "exists", "isfile", "isdir", "basename", "dirname"]:
            if hasattr(os.path, func_name):
                setattr(restricted_os.path, func_name, getattr(os.path, func_name))

        # Add path functions directly to os for convenience
        for func_name in ["join", "exists", "isfile", "isdir", "basename", "dirname"]:
            if hasattr(os.path, func_name):
                setattr(restricted_os, func_name, getattr(os.path, func_name))

        # Set up path attribute correctly
        restricted_os.path.path = restricted_os.path

        restricted_globals["os"] = restricted_os

        # Add other allowed modules with enhanced security checks
        for module_name in allowed_modules:
            if module_name not in (
                "os",
                "os.path",
            ):  # Skip os since we handled it specially
                try:
                    module = importlib.import_module(module_name)
                    # Add security check for module path
                    if hasattr(module, "__file__"):
                        module_path = Path(module.__file__)
                        if not any(
                            str(module_path).startswith(str(p)) for p in sys.path
                        ):
                            logger.warning(
                                f"Skipping module {module_name} due to path security check"
                            )
                            continue
                    restricted_globals[module_name.split(".")[0]] = module
                except ImportError as e:
                    logger.warning(f"Failed to import module {module_name}: {e}")

        # Add inputs and output_dir if provided (with deep copy for safety)
        if inputs is not None:
            restricted_globals["inputs"] = copy.deepcopy(inputs)
        if output_dir is not None:
            restricted_globals["output_dir"] = str(output_dir)

        # Final security validation
        self._validate_globals(restricted_globals)

        return restricted_globals

    def execute_in_sandbox(
        self,
        script: str,
        inputs: Dict[str, Any],
        output_dir: str,
        function_name: str = None,
        globals_dict: Dict[str, Any] = None,
    ) -> Any:
        """Execute code in a restricted sandbox environment.

        Args:
            script: Python code to execute
            inputs: Dictionary of input variables
            output_dir: Directory for file operations
            function_name: Optional function name to call
            globals_dict: Optional globals dictionary

        Returns:
            Result of execution

        Raises:
            RestrictedExecutionError: If code violates security rules
            SecurityViolationError: If a security violation is detected
            RestrictedImportError: If an unauthorized import is attempted
        """
        try:
            logger.info("Compiling code with RestrictedPython")
            byte_code = compile_restricted(
                script,
                filename="<string>",
                mode="exec",
                policy=None,
            )

            # Prepare restricted globals
            logger.info("Preparing restricted globals")
            restricted_globals = self.prepare_restricted_globals(
                inputs=inputs,
                output_dir=output_dir,
                existing_globals=globals_dict,
            )

            # Validate globals have required guards
            self._validate_globals(restricted_globals)

            # Store initial globals to compare against later
            initial_globals = set(restricted_globals.keys())

            # Execute the code
            logger.info("Executing code in sandbox")
            exec(byte_code, restricted_globals)

            # If no function name specified, look for main
            target_function = function_name or "main"

            logger.info(f"Calling function: {target_function}")
            if target_function not in restricted_globals:
                # Try to find any user-defined function
                # Only consider functions that weren't in the initial globals
                available_functions = [
                    name
                    for name, obj in restricted_globals.items()
                    if callable(obj)
                    and not name.startswith("_")
                    and name not in initial_globals
                ]
                if len(available_functions) == 1:
                    target_function = available_functions[0]
                else:
                    raise RestrictedExecutionError(
                        f"Function {target_function} not found",
                        script,
                        {"error": f"Function {target_function} not found"},
                    )

            # Call the function with appropriate arguments
            func = restricted_globals[target_function]

            if target_function == "main":
                # Main function expects specific arguments
                result = func(inputs, output_dir)

                # Validate result type for main function
                if not isinstance(result, dict):
                    raise RestrictedExecutionError(
                        "Main function must return a dictionary",
                        script,
                        {"error": "Invalid return type"},
                    )

                # Validate result values for main function
                for key, value in result.items():
                    if not isinstance(
                        value, (str, int, float, bool, list, dict, tuple)
                    ):
                        raise SecurityViolationError(
                            f"Result '{key}' has unsupported type: {type(value)}"
                        )
            else:
                # For other functions, validate input types based on AST
                import ast

                tree = ast.parse(script)
                annotations = {}
                return_annotation = None

                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name == target_function
                    ):
                        # Get type annotations from function definition
                        for arg in node.args.args:
                            if arg.annotation:
                                # Convert annotation AST to a string representation
                                if isinstance(arg.annotation, ast.Subscript):
                                    # Handle generic types like List[int]
                                    if isinstance(arg.annotation.value, ast.Name):
                                        container = arg.annotation.value.id
                                        if container == "List" and isinstance(
                                            arg.annotation.slice, ast.Name
                                        ):
                                            elem_type = arg.annotation.slice.id
                                            annotations[arg.arg] = (list, elem_type)
                                else:
                                    # Handle simple types like int, str
                                    if isinstance(arg.annotation, ast.Name):
                                        annotations[arg.arg] = arg.annotation.id

                        # Get return type annotation
                        if node.returns:
                            if isinstance(node.returns, ast.Subscript):
                                # Handle generic return types like List[int]
                                if isinstance(node.returns.value, ast.Name):
                                    container = node.returns.value.id
                                    if container == "List" and isinstance(
                                        node.returns.slice, ast.Name
                                    ):
                                        elem_type = node.returns.slice.id
                                        return_annotation = (list, elem_type)
                            else:
                                # Handle simple return types like int, str
                                if isinstance(node.returns, ast.Name):
                                    return_annotation = node.returns.id

                # Validate inputs against annotations
                for param_name, param_value in inputs.items():
                    if param_name in annotations:
                        annotation = annotations[param_name]
                        if isinstance(annotation, tuple):
                            # Handle generic types like List[int]
                            container_type, elem_type = annotation
                            if not isinstance(param_value, container_type):
                                raise TypeError(
                                    f"Parameter '{param_name}' must be a {container_type.__name__}"
                                )
                            # Check element types
                            elem_type_class = restricted_globals.get(
                                elem_type
                            ) or __builtins__.get(elem_type)
                            if elem_type_class and not all(
                                isinstance(x, elem_type_class) for x in param_value
                            ):
                                raise TypeError(
                                    f"All elements in '{param_name}' must be of type {elem_type}"
                                )
                        else:
                            # Handle simple types
                            type_class = restricted_globals.get(
                                annotation
                            ) or __builtins__.get(annotation)
                            if type_class and not isinstance(param_value, type_class):
                                # Allow numeric type coercion (int -> float)
                                if type_class == float and isinstance(param_value, int):
                                    # Convert the input to float
                                    inputs[param_name] = float(param_value)
                                else:
                                    raise TypeError(
                                        f"Parameter '{param_name}' must be of type {annotation}"
                                    )

                # Call function with appropriate arguments
                if isinstance(inputs, (list, tuple)):
                    result = func(*inputs)
                else:
                    result = func(**inputs)

                # Validate return type
                if return_annotation:
                    if isinstance(return_annotation, tuple):
                        # Handle generic return types like List[int]
                        container_type, elem_type = return_annotation
                        if not isinstance(result, container_type):
                            raise TypeError(
                                f"Return value must be a {container_type.__name__}"
                            )
                        # Check element types
                        elem_type_class = restricted_globals.get(
                            elem_type
                        ) or __builtins__.get(elem_type)
                        if elem_type_class and not all(
                            isinstance(x, elem_type_class) for x in result
                        ):
                            # Allow numeric type coercion for list elements
                            if elem_type_class == float and all(
                                isinstance(x, (int, float)) for x in result
                            ):
                                pass
                            else:
                                raise TypeError(
                                    f"All elements in return value must be of type {elem_type}"
                                )
                    else:
                        # Handle simple return types
                        type_class = restricted_globals.get(
                            return_annotation
                        ) or __builtins__.get(return_annotation)
                        if type_class and not isinstance(result, type_class):
                            # Allow numeric type coercion (int -> float)
                            if type_class == float and isinstance(result, int):
                                pass
                            else:
                                raise TypeError(
                                    f"Return value must be of type {return_annotation}"
                                )

            return result

        except (SecurityViolationError, RestrictedImportError) as e:
            # Re-raise security-related errors directly
            raise

        except Exception as e:
            tb = traceback.format_exc()
            raise RestrictedExecutionError(
                f"Execution failed: {str(e)}", script, {"traceback": tb}
            ) from e
