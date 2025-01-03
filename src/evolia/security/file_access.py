import logging
import re
import stat
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union


class FileAccessViolationError(Exception):
    """Raised when file access violates security restrictions"""

    def __init__(self, message: str):
        super().__init__(f"File access violation: {message}")


def _split_with_escaped_spaces(text: str) -> List[str]:
    """Split text into tokens, preserving escaped spaces.

    Args:
        text: Text to split

    Returns:
        List of tokens
    """
    tokens = []
    current_token = []
    escaped = False

    for char in text:
        if escaped:
            current_token.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char.isspace() and not escaped:
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
        else:
            current_token.append(char)

    if current_token:
        tokens.append("".join(current_token))

    return tokens


def extract_paths(task_description: str) -> List[str]:
    r"""Extract file and folder paths from a task description.
    Paths must be preceded by "@" to be recognized.
    Spaces in paths must be escaped with backslash (e.g. @/path/to/my\ file.txt)

    Args:
        task_description: The task description to parse for paths

    Returns:
        List of normalized paths found in the description
    """
    logger = logging.getLogger("evolia")
    paths = set()

    # Process the string character by character to handle escaped spaces
    tokens = []
    current_token = []
    in_token = False
    escaped = False

    for i, char in enumerate(task_description):
        if char == "@" and (i == 0 or task_description[i - 1].isspace()):
            # Start of new token
            if current_token:
                tokens.append("".join(current_token))
            current_token = ["@"]
            in_token = True
            escaped = False
        elif in_token:
            if escaped:
                # Previous char was backslash, add current char as is
                current_token.append(char)
                escaped = False
            elif char == "\\":
                # Current char is backslash, mark as escaped
                escaped = True
            elif char.isspace() and not escaped:
                # Unescaped space, end current token
                tokens.append("".join(current_token))
                current_token = []
                in_token = False
            else:
                # Regular character
                current_token.append(char)

    # Add final token if exists
    if current_token:
        tokens.append("".join(current_token))

    # Process tokens
    for token in tokens:
        if not token.startswith("@"):
            continue

        # Remove @ prefix and clean the path
        path = token[1:].strip()

        try:
            # Use pathlib to normalize and validate the path
            path_obj = Path(path)

            # Basic validation checks
            parts = path_obj.parts

            # Skip if path is just dots or slashes
            if str(path_obj) in [".", "..", "/"]:
                continue

            # Skip if any part contains invalid characters
            invalid_chars = {"*", "?", ">", "<", "|", "\0"}
            if any(any(c in part for c in invalid_chars) for part in parts):
                continue

            # Skip if path has double slashes or ends with . or ..
            if str(path_obj).endswith("/.") or str(path_obj).endswith("/.."):
                continue

            # Skip if path is just a file extension
            if path.startswith(".") and "/" not in path:
                continue

            # Add normalized path
            paths.add(str(path_obj))

        except (ValueError, OSError) as e:
            logger.warning(
                "Invalid path in task description: {path}",
                extra={
                    "payload": {
                        "path": path,
                        "component": "file_access",
                        "operation": "path_extraction",
                    }
                },
                exc_info=True,
            )
            continue

    return sorted(list(paths))


def _check_path_exists(path: Union[str, Path], log_warning: bool = True) -> bool:
    """Check if a path exists.

    Args:
        path: Path to check
        log_warning: Whether to log a warning if the path doesn't exist

    Returns:
        bool: True if path exists, False otherwise
    """
    logger = logging.getLogger("evolia")
    path_obj = Path(path)
    exists = path_obj.exists()

    if not exists and log_warning:
        logger.warning(
            f"Path does not exist: {path}",
            extra={
                "payload": {
                    "path": path,
                    "component": "file_access",
                    "operation": "path_check",
                }
            },
        )

    return exists


def get_allowed_paths(
    config_permissions: List[Dict[str, Any]], cli_permissions: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Get allowed paths by merging config and CLI permissions.

    Args:
        config_permissions: List of permission configurations from config file
        cli_permissions: Permissions from CLI arguments

    Returns:
        Dict mapping permission types to lists of allowed paths and path configs
    """
    logger = logging.getLogger("evolia")
    permissions = {
        "read": [],
        "write": [],
        "create": [],
        "create_only": [],
        "path_configs": [],  # Store full path configurations for special flags like append_only
    }

    # Process config permissions
    for path_config in config_permissions:
        path = path_config.get("path")
        access = path_config.get("access", {})
        recursive = path_config.get("recursive", False)
        append_only = path_config.get("append_only", False)

        if not path:
            continue

        try:
            # Check if path exists before resolving
            if not _check_path_exists(path):
                continue

            # Use pathlib to resolve path
            resolved_path = Path(path).resolve()

            # If recursive, add the path and all its subdirectories
            if recursive:
                paths_to_add = [resolved_path]
                # Use pathlib's rglob instead of os.walk
                paths_to_add.extend(
                    [d.resolve() for d in resolved_path.rglob("*") if d.is_dir()]
                )
            else:
                paths_to_add = [resolved_path]

            # Add permissions based on access settings
            for p in paths_to_add:
                if access.get("read"):
                    permissions["read"].append(str(p))
                if access.get("write"):
                    permissions["write"].append(str(p))
                if access.get("create"):
                    permissions["create"].append(str(p))
                if access.get("create_only"):
                    permissions["create_only"].append(str(p))

                # Store full path config if it has special flags
                if append_only:
                    permissions["path_configs"].append(
                        {"path": str(p), "append_only": True, "access": access}
                    )

        except Exception as e:
            logger.warning(
                f"Invalid path in config permissions: {path}",
                extra={
                    "payload": {
                        "path": path,
                        "component": "file_access",
                        "operation": "config_permissions",
                    }
                },
                exc_info=True,
            )
            continue

    # Process CLI permissions (these override config)
    if cli_permissions:
        for perm_type in ["read", "write", "create", "create_only"]:
            paths = cli_permissions.get(perm_type, [])
            validated_paths = []

            for path in paths:
                try:
                    # Check if path exists before resolving
                    if not _check_path_exists(path):
                        continue

                    # Use pathlib to resolve path
                    resolved_path = Path(path).resolve()
                    validated_paths.append(str(resolved_path))
                except Exception as e:
                    logger.warning(
                        f"Invalid path in CLI permissions: {path}",
                        extra={
                            "payload": {
                                "path": path,
                                "component": "file_access",
                                "operation": "cli_permissions",
                            }
                        },
                        exc_info=True,
                    )
                    continue

            # CLI permissions override config permissions
            if validated_paths:
                permissions[perm_type] = validated_paths

    # Remove duplicates while preserving order
    for perm_type in ["read", "write", "create", "create_only"]:
        permissions[perm_type] = list(dict.fromkeys(permissions[perm_type]))

    # Log final permissions
    logger.debug(
        "Resolved file access permissions",
        extra={
            "payload": {
                "permissions": permissions,
                "component": "file_access",
                "operation": "resolve_permissions",
            }
        },
    )

    return permissions


def get_base_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Get base paths from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of base paths with defaults
    """
    paths_config = config.get("file_access", {}).get("paths", {})
    return {
        "ephemeral_base": paths_config.get("ephemeral_base", "run_artifacts"),
        "tools_base": paths_config.get("tools_base", "tools/system"),
        "data_base": paths_config.get("data_base", "data"),
        "logs_base": paths_config.get("logs_base", "logs"),
    }


def resolve_path_template(path: str, base_paths: Dict[str, str]) -> str:
    """Resolve a path template using base paths.

    Args:
        path: Path template that may contain ${paths.X} references
        base_paths: Dictionary of base paths

    Returns:
        Resolved path string
    """
    for key, value in base_paths.items():
        path = path.replace(f"${{paths.{key}}}", value)
    return path


def _check_read_permission(
    file_path: Path, permissions: Dict[str, List[str]], default_policy: str = "deny"
) -> None:
    """Check if reading from a path is allowed.

    Args:
        file_path: Path to validate
        permissions: Dictionary of permission lists
        default_policy: Default policy when path is not explicitly allowed

    Raises:
        FileAccessViolationError: If read access is denied
    """
    logger = logging.getLogger("evolia")
    if permissions is None:
        permissions = {}

    resolved_file_path = Path(file_path).resolve()
    read_paths = [Path(p).resolve() for p in permissions.get("read", [])]
    is_allowed = any(
        str(resolved_file_path).startswith(str(allowed_path))
        for allowed_path in read_paths
    )

    # If not explicitly allowed, check default policy
    if not is_allowed and default_policy == "allow":
        is_allowed = True
        logger.debug(
            f"Path {file_path} allowed by default policy",
            extra={
                "payload": {
                    "path": str(file_path),
                    "component": "file_access",
                    "operation": "read_permission",
                    "default_policy": default_policy,
                }
            },
        )

    if not is_allowed:
        raise FileAccessViolationError(
            f"Reading from {file_path} denied. Must be within allowed paths: {read_paths}"
        )


def _check_write_permission(
    file_path: Path,
    permissions: Dict[str, List[str]],
    ephemeral_dir: Optional[str] = None,
) -> None:
    """Check if writing to a path is allowed.

    Args:
        file_path: Path to validate
        permissions: Dictionary of permission lists
        ephemeral_dir: Optional ephemeral directory for write operations

    Raises:
        FileAccessViolationError: If write access is denied
    """
    logger = logging.getLogger("evolia")
    if permissions is None:
        permissions = {}

    resolved_file_path = Path(file_path).resolve()
    write_paths = [Path(p).resolve() for p in permissions.get("write", [])]
    create_paths = [Path(p).resolve() for p in permissions.get("create", [])]
    create_only_paths = [Path(p).resolve() for p in permissions.get("create_only", [])]

    # Add ephemeral directory to write paths if provided
    if ephemeral_dir:
        ephemeral_path = Path(ephemeral_dir).resolve()
        if str(ephemeral_path) not in [str(p) for p in write_paths]:
            write_paths.append(ephemeral_path)

    # Check if we're trying to modify an existing file
    file_exists = resolved_file_path.exists()
    if file_exists:
        # For existing files, check write permissions
        is_allowed = any(
            str(resolved_file_path).startswith(str(allowed_path))
            for allowed_path in write_paths
        )

        if not is_allowed:
            if ephemeral_dir:
                logger.warning(
                    "Write attempt outside ephemeral directory",
                    extra={
                        "payload": {
                            "path": str(file_path),
                            "ephemeral_dir": ephemeral_dir,
                            "component": "file_access",
                            "operation": "write_permission",
                        }
                    },
                )
            raise FileAccessViolationError(
                f"Writing to {file_path} denied. Must be within allowed paths: {write_paths}"
            )

        # Check if we're in a creation-only directory
        is_creation_only = any(
            str(resolved_file_path).startswith(str(allowed_path))
            for allowed_path in create_only_paths
        )
        if is_creation_only:
            raise FileAccessViolationError(
                f"Directory is creation-only, cannot modify existing files: {file_path}"
            )
    else:
        # For new files, check create permissions
        is_allowed = any(
            str(resolved_file_path).startswith(str(allowed_path))
            for allowed_path in (write_paths + create_paths + create_only_paths)
        )

        if not is_allowed:
            if ephemeral_dir:
                logger.warning(
                    "Create attempt outside allowed paths",
                    extra={
                        "payload": {
                            "path": str(file_path),
                            "ephemeral_dir": ephemeral_dir,
                            "component": "file_access",
                            "operation": "create_permission",
                        }
                    },
                )
            raise FileAccessViolationError(
                f"Creating file at {file_path} denied. Must be within allowed paths: {write_paths + create_paths + create_only_paths}"
            )

        # Check if parent directory exists
        parent_dir = resolved_file_path.parent
        if not parent_dir.exists():
            raise FileAccessViolationError(
                f"Parent directory does not exist: {parent_dir}"
            )


def _check_create_permission(
    file_path: Path,
    permissions: Dict[str, List[str]],
    ephemeral_dir: Optional[str] = None,
) -> None:
    """Check if creating a file at a path is allowed.

    Args:
        file_path: Path to validate
        permissions: Dictionary of permission lists
        ephemeral_dir: Optional ephemeral directory for write operations

    Raises:
        FileAccessViolationError: If create access is denied
    """
    logger = logging.getLogger("evolia")
    if permissions is None:
        permissions = {}

    resolved_file_path = Path(file_path).resolve()
    create_paths = [Path(p).resolve() for p in permissions.get("create", [])]

    # Add ephemeral directory to create paths if provided
    if ephemeral_dir:
        ephemeral_path = Path(ephemeral_dir).resolve()
        if str(ephemeral_path) not in [str(p) for p in create_paths]:
            create_paths.append(ephemeral_path)

    # Check if parent directory exists and is writable
    parent_dir = resolved_file_path.parent
    if not parent_dir.exists():
        raise FileAccessViolationError(f"Parent directory does not exist: {parent_dir}")

    is_allowed = any(
        str(resolved_file_path).startswith(str(allowed_path))
        for allowed_path in create_paths
    )

    if not is_allowed:
        if ephemeral_dir:
            logger.warning(
                "Create attempt outside ephemeral directory",
                extra={
                    "payload": {
                        "path": str(file_path),
                        "ephemeral_dir": ephemeral_dir,
                        "component": "file_access",
                        "operation": "create_permission",
                    }
                },
            )
        raise FileAccessViolationError(
            f"Creating file at {file_path} denied. Must be within allowed paths: {create_paths}"
        )


def validate_path(
    file_path: Union[str, Path],
    allowed_paths: Optional[List[str]] = None,
    mode: str = "r",
    ephemeral_dir: Optional[str] = None,
    permissions: Optional[Dict[str, List[str]]] = None,
    default_policy: str = "deny",
) -> Path:
    """Validate and resolve a file path based on permissions.

    Args:
        file_path: Path to validate
        allowed_paths: DEPRECATED - do not use, will be removed
        mode: File open mode ('r', 'w', 'a')
        ephemeral_dir: Optional ephemeral directory for write operations
        permissions: Dictionary of permission lists (required)
        default_policy: Default policy when path is not explicitly allowed

    Returns:
        Resolved Path object

    Raises:
        FileAccessViolationError: If access is denied
    """
    logger = logging.getLogger("evolia")

    # Deprecation error for legacy system
    if allowed_paths is not None:
        raise ValueError(
            "The allowed_paths parameter is deprecated and will be removed. "
            "Please migrate to the permissions system."
        )

    # Require permissions
    if permissions is None:
        raise ValueError("The permissions parameter is required")

    # Check for path traversal before resolving
    if ".." in Path(file_path).parts:
        raise FileAccessViolationError(f"Path traversal detected: {file_path}")

    # Convert to Path and resolve
    try:
        resolved_file_path = Path(file_path).resolve(strict=False)
    except RuntimeError:
        raise FileAccessViolationError(f"Invalid path: {file_path}")

    # Resolve ephemeral path if provided
    ephemeral_path = None
    if ephemeral_dir:
        try:
            ephemeral_path = Path(ephemeral_dir).resolve(strict=False)
        except RuntimeError:
            raise FileAccessViolationError(
                f"Invalid ephemeral directory: {ephemeral_dir}"
            )

    # Check if this is a write operation
    is_write = "w" in mode or "a" in mode or "+" in mode

    # For read operations, check if file exists
    if not is_write and not resolved_file_path.exists():
        raise FileAccessViolationError(f"File does not exist: {file_path}")

    # For write operations, check parent directory first
    if is_write:
        parent_dir = resolved_file_path.parent
        if not parent_dir.exists():
            raise FileAccessViolationError(
                f"Parent directory does not exist: {parent_dir}"
            )

    # Check if we're in a creation-only directory first
    if is_write and "create_only" in permissions:
        is_creation_only = any(
            str(resolved_file_path).startswith(str(Path(p).resolve()))
            for p in permissions["create_only"]
        )
        if is_creation_only and resolved_file_path.exists():
            raise FileAccessViolationError(
                f"Directory is creation-only, cannot modify existing files: {file_path}"
            )

    # Then check basic permissions
    if is_write:
        _check_write_permission(resolved_file_path, permissions, ephemeral_dir)
    else:
        _check_read_permission(resolved_file_path, permissions, default_policy)

    # Finally check ephemeral directory restrictions
    if is_write and ephemeral_dir:
        if (
            str(resolved_file_path).startswith(str(ephemeral_path))
            and resolved_file_path.exists()
        ):
            # Only prevent modification if we don't have explicit write permission
            write_paths = [Path(p).resolve() for p in permissions.get("write", [])]
            has_write_permission = any(
                str(resolved_file_path).startswith(str(allowed_path))
                for allowed_path in write_paths
            )
            if not has_write_permission:
                raise FileAccessViolationError(
                    "Cannot modify existing file in ephemeral directory"
                )

    return resolved_file_path


def restricted_open(
    file_path: str,
    mode: str = "r",
    allowed_paths: Optional[List[str]] = None,
    ephemeral_dir: Optional[str] = None,
    permissions: Optional[Dict[str, List[str]]] = None,
    default_policy: str = "deny",
) -> IO:
    """Open a file with restricted access.

    Args:
        file_path: Path to file
        mode: File mode ('r', 'w', 'a')
        allowed_paths: DEPRECATED - do not use, will be removed
        ephemeral_dir: Optional ephemeral directory for write operations
        permissions: Dictionary mapping permission types to lists of allowed paths (required)
        default_policy: Default policy when path is not explicitly allowed

    Returns:
        File object

    Raises:
        FileAccessViolationError: If file access is denied
    """
    logger = logging.getLogger("evolia")

    # Validate mode string
    if not re.match(r"^[rwa]\+?$", mode):
        raise FileAccessViolationError(
            f"Invalid mode: {mode}. Only r, w, a modes are allowed"
        )

    # Check if write operation without ephemeral directory
    if ("w" in mode or "a" in mode or "+" in mode) and not ephemeral_dir:
        logger.warning(
            "Write attempt without ephemeral directory specified",
            extra={
                "payload": {
                    "path": str(file_path),
                    "mode": mode,
                    "component": "file_access",
                    "operation": "open",
                }
            },
        )
        raise FileAccessViolationError(
            "Write operations require an ephemeral directory"
        )

    try:
        validated_path = validate_path(
            file_path,
            mode=mode,
            ephemeral_dir=ephemeral_dir,
            permissions=permissions,
            default_policy=default_policy,
        )

        # Create parent directories if writing
        if "w" in mode or "a" in mode or "+" in mode:
            Path(validated_path).parent.mkdir(parents=True, exist_ok=True)

        return open(validated_path, mode)
    except FileAccessViolationError:
        raise
    except Exception as e:
        logger.error(f"Error opening file: {str(e)}", exc_info=True)
        raise FileAccessViolationError(f"Error opening file: {str(e)}")


def get_safe_open(
    allowed_paths: List[str] = None,
    ephemeral_dir: str = None,
    permissions: Dict[str, List[str]] = None,
    default_policy: str = "deny",
):
    """
    Returns a partially applied restricted_open function with preset paths and permissions.

    Args:
        allowed_paths: List of paths that are allowed to be accessed (deprecated, use permissions instead)
        ephemeral_dir: Optional ephemeral directory for write operations
        permissions: Dictionary mapping permission types ('read', 'write', 'create') to lists of allowed paths
        default_policy: Default policy when path is not explicitly allowed ('deny' or 'allow')

    Returns:
        function: A restricted open function with preset paths and permissions
    """
    logger = logging.getLogger("evolia")
    logger.debug(
        "Creating safe_open function",
        extra={
            "payload": {
                "allowed_paths": allowed_paths,
                "permissions": permissions,
                "ephemeral_dir": ephemeral_dir,
                "default_policy": default_policy,
            }
        },
    )

    def safe_open(filename: str, mode: str = "r") -> IO:
        return restricted_open(
            filename,
            mode,
            allowed_paths=allowed_paths,
            ephemeral_dir=ephemeral_dir,
            permissions=permissions,
            default_policy=default_policy,
        )

    return safe_open
