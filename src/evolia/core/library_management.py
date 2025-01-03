"""Library management module for Evolia."""

import os
import sys
import ast
import json
import time
import shlex
import logging
import subprocess
import importlib
import importlib.util
import importlib.metadata
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from pathlib import Path
from packaging.version import Version
from packaging.requirements import Requirement

from ..utils.logger import setup_logger
from ..utils.exceptions import SecurityViolationError

logger = setup_logger()


class ImportValidationError(Exception):
    """Raised when import validation fails"""

    pass


def validate_imports(imports: Set[str], config: Dict[str, Any]) -> List[str]:
    """Validate a set of imports against configuration.

    Args:
        imports: Set of import names to validate
        config: Configuration dictionary with allowed/blocked modules

    Returns:
        List of validation error messages
    """
    validator = ImportValidator(config)
    return validator.validate_imports(imports)


class ImportValidator:
    """Validates imports against configuration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allowed_modules = config.get("allowed_modules", {})
        self.blocked_modules = set(
            config.get("library_management", {}).get("blocked_libraries", [])
        )

    def validate_imports(self, imports: Set[str]) -> List[str]:
        """Validate a set of imports.

        Args:
            imports: Set of import names to validate

        Returns:
            List of validation error messages
        """
        errors = []

        for imp in imports:
            # Check if module is blocked
            if imp in self.blocked_modules:
                errors.append(f"Module {imp} is blocked by configuration")
                continue

            # Check if module is allowed
            if imp not in self.allowed_modules and imp not in STDLIB_MODULES:
                errors.append(f"Module {imp} is not in allowed modules list")
                continue

            # Check version if specified
            if imp in self.allowed_modules and self.allowed_modules[imp]:
                valid, version = validate_library_version(
                    imp, self.allowed_modules[imp]
                )
                if not valid:
                    errors.append(
                        f"Module {imp} version {version} does not meet minimum requirement {self.allowed_modules[imp]}"
                    )

        return errors


# Standard library modules that don't need version checks
STDLIB_MODULES = {
    "abc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "base64",
    "bisect",
    "calendar",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "fileinput",
    "fnmatch",
    "functools",
    "glob",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "pprint",
    "queue",
    "re",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "zipfile",
}


def detect_missing_libraries(required_libraries: List[str]) -> List[str]:
    """Detect which required libraries are missing from the environment.

    Args:
        required_libraries: List of library names to check

    Returns:
        List of missing library names
    """
    missing = []
    for lib in required_libraries:
        # Skip standard library modules
        if lib in STDLIB_MODULES:
            continue

        try:
            # Try to find the library spec first
            spec = importlib.util.find_spec(lib)
            if spec is None:
                missing.append(lib)
                continue

            # Try to get metadata to verify it's properly installed
            importlib.metadata.metadata(lib)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            missing.append(lib)
            logger.warning(
                f"Library {lib} not found",
                extra={
                    "payload": {
                        "library": lib,
                        "component": "library_management",
                        "operation": "detect_missing",
                    }
                },
            )
    return missing


def validate_library_version(
    library: str, min_version: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Validate that a library is installed and meets minimum version requirements.

    Args:
        library: Name of the library to validate
        min_version: Minimum required version string (optional)

    Returns:
        Tuple of (is_valid, installed_version)
    """
    try:
        installed_version = importlib.metadata.version(library)
        logger.debug(
            f"Checking version for {library}",
            extra={
                "payload": {
                    "library": library,
                    "installed_version": installed_version,
                    "minimum_version": min_version,
                    "component": "library_management",
                    "operation": "validate_version",
                    "validation_stage": "version_check",
                }
            },
        )

        if min_version:
            try:
                installed = Version(installed_version)
                required = Version(min_version)
                is_valid = installed >= required

                log_level = logging.WARNING if not is_valid else logging.DEBUG
                logger.log(
                    log_level,
                    f"Version validation for {library}: {'FAILED' if not is_valid else 'PASSED'}",
                    extra={
                        "payload": {
                            "library": library,
                            "installed_version": str(installed),
                            "minimum_version": str(required),
                            "is_valid": is_valid,
                            "component": "library_management",
                            "operation": "validate_version",
                            "validation_stage": "version_comparison",
                            "validation_details": {
                                "installed_version_parsed": str(installed),
                                "minimum_version_parsed": str(required),
                                "comparison_result": "version_too_low"
                                if not is_valid
                                else "version_ok",
                            },
                        }
                    },
                )
                return is_valid, installed_version

            except ValueError as e:
                logger.error(
                    f"Invalid version format for {library}",
                    extra={
                        "payload": {
                            "library": library,
                            "installed_version": installed_version,
                            "minimum_version": min_version,
                            "error": str(e),
                            "component": "library_management",
                            "operation": "validate_version",
                            "validation_stage": "version_parsing",
                            "validation_details": {
                                "error_type": "version_parse_error",
                                "error_message": str(e),
                            },
                        }
                    },
                )
                return False, installed_version

        # If no minimum version specified, just check if installed
        logger.debug(
            f"No version requirement for {library}",
            extra={
                "payload": {
                    "library": library,
                    "installed_version": installed_version,
                    "component": "library_management",
                    "operation": "validate_version",
                    "validation_stage": "completion",
                    "validation_details": {"status": "no_version_requirement"},
                }
            },
        )
        return True, installed_version

    except importlib.metadata.PackageNotFoundError:
        logger.error(
            f"Package {library} not found",
            extra={
                "payload": {
                    "library": library,
                    "error": "package_not_found",
                    "component": "library_management",
                    "operation": "validate_version",
                    "validation_stage": "package_check",
                    "validation_details": {
                        "error_type": "package_not_found",
                        "status": "missing",
                    },
                }
            },
        )
        return False, None


def get_library_dependencies(library: str) -> Set[str]:
    """Get the dependencies of a library.

    Args:
        library: Name of the library to check

    Returns:
        Set of dependency names with their version requirements
    """
    try:
        metadata = importlib.metadata.metadata(library)
        requires = metadata.get_all("Requires-Dist") or []
        dependencies = set()

        for req in requires:
            if req is None:
                continue

            try:
                # Parse requirement string using packaging.requirements
                requirement = Requirement(req)

                # Skip environment markers that don't match current environment
                if requirement.marker and not requirement.marker.evaluate():
                    continue

                # Add the package name to dependencies
                dependencies.add(requirement.name)

                logger.debug(
                    f"Found dependency for {library}",
                    extra={
                        "payload": {
                            "library": library,
                            "dependency": requirement.name,
                            "specifier": str(requirement.specifier)
                            if requirement.specifier
                            else None,
                            "marker": str(requirement.marker)
                            if requirement.marker
                            else None,
                            "component": "library_management",
                        }
                    },
                )

            except Exception as e:
                logger.warning(
                    f"Failed to parse requirement '{req}' for {library}",
                    extra={
                        "payload": {
                            "library": library,
                            "requirement": req,
                            "error": str(e),
                            "component": "library_management",
                        }
                    },
                )
                # Fall back to simple name extraction for invalid requirements
                try:
                    # Take everything up to the first space or comparison operator
                    simple_name = (
                        req.split(" ")[0].split(">=")[0].split("<=")[0].split("==")[0]
                    )
                    dependencies.add(simple_name)
                    logger.debug(
                        f"Extracted simple name for invalid requirement",
                        extra={
                            "payload": {
                                "library": library,
                                "requirement": req,
                                "extracted_name": simple_name,
                                "component": "library_management",
                            }
                        },
                    )
                except Exception as e2:
                    logger.error(
                        f"Failed to extract simple name from requirement '{req}'",
                        extra={
                            "payload": {
                                "library": library,
                                "requirement": req,
                                "error": str(e2),
                                "component": "library_management",
                            }
                        },
                    )

        return dependencies

    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            f"Package {library} not found",
            extra={"payload": {"library": library, "component": "library_management"}},
        )
        return set()
    except Exception as e:
        logger.error(
            f"Error getting dependencies for {library}: {str(e)}",
            extra={
                "payload": {
                    "library": library,
                    "error": str(e),
                    "component": "library_management",
                }
            },
        )
        return set()


class LibraryManager:
    """Manages library access and validation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allowed_libraries = set()
        self.library_versions = {}

        # Initialize from config
        allowed_modules = config.get("allowed_modules", {})
        if isinstance(allowed_modules, dict):
            # New format with version requirements
            for lib, version in allowed_modules.items():
                self.allowed_libraries.add(lib)
                if version is not None:
                    self.library_versions[lib] = version
        else:
            # Old format (list of library names)
            self.allowed_libraries.update(allowed_modules)

        # Add standard library modules
        self.allowed_libraries.update(STDLIB_MODULES)

        # Add runtime overrides
        runtime_overrides = config.get("library_management", {}).get(
            "runtime_overrides", {}
        )
        for lib in runtime_overrides.get("allowed", []):
            if isinstance(lib, dict):
                self.add_allowed_library(lib["name"], lib.get("version"))
            else:
                self.add_allowed_library(lib)

        # Remove blocked libraries
        blocked = set(config.get("library_management", {}).get("blocked_libraries", []))
        blocked.update(runtime_overrides.get("blocked", []))
        self.allowed_libraries.difference_update(blocked)

        logger.debug(
            "Library manager initialized",
            extra={
                "payload": {
                    "allowed_libraries": list(self.allowed_libraries),
                    "library_versions": self.library_versions,
                    "blocked_libraries": list(blocked),
                    "component": "library_management",
                    "operation": "init",
                }
            },
        )

    def validate_imports(self, imports: Set[str]) -> List[str]:
        """Validate a set of imports against allowed libraries.

        Args:
            imports: Set of module names to validate

        Returns:
            List of validation error messages
        """
        errors = []
        for module in imports:
            if module not in self.allowed_libraries:
                msg = f"Unauthorized module import: {module}"
                errors.append(msg)
                logger.warning(
                    msg,
                    extra={
                        "payload": {
                            "module": module,
                            "allowed_modules": list(self.allowed_libraries),
                            "component": "library_management",
                            "operation": "validate_imports",
                        }
                    },
                )

            elif module in self.library_versions:
                is_valid, version = validate_library_version(
                    module, self.library_versions[module]
                )
                if not is_valid:
                    msg = f"Module {module} version {version} does not meet minimum requirement: {self.library_versions[module]}"
                    errors.append(msg)
                    logger.warning(
                        msg,
                        extra={
                            "payload": {
                                "module": module,
                                "installed_version": version,
                                "required_version": self.library_versions[module],
                                "component": "library_management",
                                "operation": "validate_imports",
                            }
                        },
                    )
        return errors

    def check_dependencies(self, libraries: Set[str]) -> List[str]:
        """Check if all dependencies of the given libraries are allowed.

        Args:
            libraries: Set of library names to check

        Returns:
            List of disallowed dependency names
        """
        all_deps = set()
        for lib in libraries:
            deps = get_library_dependencies(lib)
            all_deps.update(deps)
            logger.debug(
                f"Found dependencies for {lib}",
                extra={
                    "payload": {
                        "library": lib,
                        "dependencies": list(deps),
                        "component": "library_management",
                        "operation": "check_dependencies",
                    }
                },
            )

        disallowed = [dep for dep in all_deps if dep not in self.allowed_libraries]
        if disallowed:
            logger.warning(
                "Found disallowed dependencies",
                extra={
                    "payload": {
                        "disallowed_dependencies": disallowed,
                        "allowed_libraries": list(self.allowed_libraries),
                        "component": "library_management",
                        "operation": "check_dependencies",
                    }
                },
            )
        return disallowed

    def add_allowed_library(
        self, library: str, min_version: Optional[str] = None
    ) -> None:
        """Add a library to the allowed list.

        Args:
            library: Name of the library to allow
            min_version: Minimum required version string (optional)
        """
        self.allowed_libraries.add(library)
        if min_version:
            self.library_versions[library] = min_version
        logger.info(
            f"Added library {library} to allowed list",
            extra={
                "payload": {
                    "library": library,
                    "min_version": min_version,
                    "component": "library_management",
                    "operation": "add_allowed_library",
                }
            },
        )
