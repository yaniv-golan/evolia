"""Restricted execution module for Evolia."""

import builtins
import importlib
import logging
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Dict, Optional, Set

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import safer_getattr

from ..utils.exceptions import EvoliaError, SecurityViolationError

logger = logging.getLogger(__name__)


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


def secure_open(filename, mode="r", *args, **kwargs):
    """Secure file open function that restricts access to output_dir."""
    if not isinstance(filename, (str, Path)):
        raise SecurityViolationError("Filename must be a string or Path object")

    path = Path(filename)
    if not path.is_absolute():
        # Get output_dir from the caller's globals
        frame = sys._getframe(1)
        while frame is not None:
            if "output_dir" in frame.f_globals:
                output_dir = frame.f_globals["output_dir"]
                break
            frame = frame.f_back
        if frame is None or not output_dir:
            raise SecurityViolationError("output_dir not found in globals")
        path = Path(output_dir) / path

    # Ensure the path is within output_dir
    try:
        path = path.resolve()
        frame = sys._getframe(1)
        while frame is not None:
            if "output_dir" in frame.f_globals:
                output_dir = Path(frame.f_globals["output_dir"]).resolve()
                break
            frame = frame.f_back
        if frame is None:
            raise SecurityViolationError("output_dir not found in globals")
        if not str(path).startswith(str(output_dir)):
            raise SecurityViolationError(
                f"Access to {path} outside output_dir is not allowed"
            )
    except Exception as e:
        raise SecurityViolationError(f"Invalid path: {str(e)}")

    # Only allow read/write operations
    if mode not in ("r", "w", "a", "r+", "w+", "a+"):
        raise SecurityViolationError(f"File mode {mode} is not allowed")

    return open(str(path), mode, *args, **kwargs)


def restricted_import(
    name, globals=None, locals=None, fromlist=(), level=0, allowed_modules=None
):
    """Restricted import function that only allows specific modules."""
    if allowed_modules is None:
        allowed_modules = set()  # Empty set by default for maximum security

    # Check if the module or any of its parents are allowed
    module_parts = name.split(".")
    for i in range(len(module_parts)):
        current_module = ".".join(module_parts[: i + 1])
        if current_module in allowed_modules:
            # Import is allowed, perform it
            try:
                # Special handling for os and os.path - use the pre-created modules
                if name == "os.path":
                    # For os.path import, return the path module from os
                    frame = sys._getframe(1)
                    while frame is not None:
                        if "os" in frame.f_globals:
                            return frame.f_globals["os"].path
                        frame = frame.f_back
                    raise ImportError("Could not find pre-created os module")
                elif name == "os":
                    frame = sys._getframe(1)
                    while frame is not None:
                        if "os" in frame.f_globals:
                            return frame.f_globals["os"]
                        frame = frame.f_back
                    raise ImportError("Could not find pre-created os module")

                # For all other modules, perform normal import
                module = __import__(name, globals, locals, fromlist, level)

                # If this is a submodule import (e.g., os.path), return the submodule
                if "." in name:
                    for part in name.split(".")[1:]:
                        module = getattr(module, part)

                return module

            except ImportError as e:
                raise RestrictedImportError(
                    f"Failed to import allowed module '{name}': {str(e)}",
                    name,
                    {"error": str(e)},
                )

    raise RestrictedImportError(
        f"Import of module '{name}' is not allowed",
        name,
        {"allowed_modules": list(allowed_modules)},
    )


class RestrictedExecutor:
    """Handles execution of code in a restricted environment."""

    def __init__(self, allowed_modules: Set[str], allowed_builtins: Set[str]):
        """Initialize with allowed modules and builtins."""
        self.allowed_modules = allowed_modules
        self.allowed_builtins = allowed_builtins
        self.logger = logging.getLogger(__name__)

    def prepare_restricted_globals(
        self, inputs=None, output_dir=None, allowed_modules=None
    ):
        """Prepare restricted globals for code execution."""
        logger.info("Adding allowed modules")

        if allowed_modules is None:
            allowed_modules = {
                "os",
                "os.path",
                "json",
                "pandas",
                "numpy",
                "math",
                "pathlib",
            }

        # Start with safe builtins from RestrictedPython
        restricted_globals = safe_builtins.copy()

        # Add our custom guarded builtins
        restricted_globals.update(
            {
                "_getattr_": safer_getattr,
                "_write_": lambda x: x,  # Allow writing to files
                "_getiter_": iter,
                "_print_": print,
                "_inplacevar_": lambda op, x, y: op(x, y),
                "__import__": lambda name, globals=None, locals=None, fromlist=(), level=0: restricted_import(
                    name,
                    globals,
                    locals,
                    fromlist,
                    level,
                    allowed_modules=allowed_modules,
                ),
                "open": secure_open,  # Use our secure file operations
                "sum": sum,  # Add missing builtins
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
            }
        )

        # Add allowed modules
        restricted_globals["__builtins__"] = restricted_globals

        # Add RestrictedPython guards
        restricted_globals["_getitem_"] = guarded_getitem
        restricted_globals["_write_"] = lambda obj: obj  # Allow all writes
        restricted_globals["_getattr_"] = safer_getattr

        # Import and add os module with restricted functionality
        import os
        import os.path

        # Create a restricted os module
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

        # Add other allowed modules
        for module_name in allowed_modules:
            if module_name not in (
                "os",
                "os.path",
            ):  # Skip os since we handled it specially
                try:
                    module = __import__(module_name)
                    restricted_globals[module_name.split(".")[0]] = module
                except ImportError as e:
                    logger.warning(f"Failed to import module {module_name}: {e}")

        # Add inputs and output_dir if provided
        if inputs is not None:
            restricted_globals["inputs"] = inputs
        if output_dir is not None:
            restricted_globals["output_dir"] = output_dir

        return restricted_globals

    def execute_in_sandbox(
        self,
        script: str,
        inputs: Dict[str, Any],
        output_dir: str,
        function_name: str = None,
    ) -> Any:
        """Execute code in a restricted sandbox environment.

        Args:
            script: The Python code to execute
            inputs: Dictionary of input variables
            output_dir: Directory for any output files
            function_name: Name of the function to execute (if None, looks for 'main')

        Returns:
            Result of the function execution

        Raises:
            RestrictedExecutionError: If execution fails or security constraints are violated
        """
        try:
            logger.info("Compiling code with RestrictedPython")
            byte_code = compile_restricted(script, filename="<string>", mode="exec")

            # Get initial globals to compare against later
            initial_globals = set(
                self.prepare_restricted_globals(
                    inputs=inputs,
                    output_dir=output_dir,
                    allowed_modules=self.allowed_modules,
                ).keys()
            )

            # Prepare restricted globals
            restricted_globals = self.prepare_restricted_globals(
                inputs=inputs,
                output_dir=output_dir,
                allowed_modules=self.allowed_modules,
            )

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

                if available_functions:
                    target_function = available_functions[0]
                    logger.info(f"Using available function: {target_function}")
                else:
                    raise RestrictedExecutionError(
                        f"No {target_function} function defined in script",
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
                # For other functions, pass inputs as individual arguments
                if isinstance(inputs, (list, tuple)):
                    result = func(*inputs)
                else:
                    result = func(**inputs)

            return result

        except SyntaxError as e:
            error_msg = f"Syntax error in script: {str(e)}"
            logger.error(error_msg)
            raise RestrictedExecutionError(error_msg, script, {"error": str(e)})

        except Exception as e:
            error_msg = f"Unexpected error executing restricted code: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise RestrictedExecutionError(error_msg, script, {"error": str(e)})
