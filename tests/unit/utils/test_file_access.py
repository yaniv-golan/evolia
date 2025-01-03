"""Tests for file access functionality"""
import os
import tempfile
from pathlib import Path

import pytest

from evolia.security.file_access import (
    FileAccessViolationError,
    extract_paths,
    get_allowed_paths,
    get_safe_open,
    restricted_open,
    validate_path,
)


def test_file_access_control():
    """Test that file access is properly restricted"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ephemeral_path = temp_path / "ephemeral"
        ephemeral_path.mkdir()

        # Create test file in allowed directory
        test_file = temp_path / "test.txt"
        with open(test_file, "w") as f:
            f.write("test content")

        permissions = {
            "read": [str(temp_path)],
            "write": [str(ephemeral_path)],
            "create": [str(ephemeral_path)],
        }

        # Test reading from allowed path
        with restricted_open(
            str(test_file),
            "r",
            permissions=permissions,
            ephemeral_dir=str(ephemeral_path),
        ) as f:
            content = f.read()
            assert "test content" in content

        # Test writing to ephemeral path
        write_file = ephemeral_path / "write.txt"
        with restricted_open(
            str(write_file),
            "w",
            permissions=permissions,
            ephemeral_dir=str(ephemeral_path),
        ) as f:
            f.write("new content")

        # Verify write worked
        with open(write_file) as f:
            assert "new content" in f.read()

        # Test access outside allowed paths
        with pytest.raises(FileAccessViolationError):
            with restricted_open(
                "/etc/passwd",
                "r",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            ):
                pass


def test_write_outside_ephemeral():
    """Test that writing outside ephemeral directory is blocked"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ephemeral_path = temp_path / "ephemeral"
        ephemeral_path.mkdir()

        # Try to write outside ephemeral directory
        outside_file = temp_path / "outside.txt"

        permissions = {
            "read": [str(temp_path)],
            "write": [str(ephemeral_path)],
            "create": [str(ephemeral_path)],
        }

        with pytest.raises(FileAccessViolationError):
            with restricted_open(
                str(outside_file),
                "w",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            ):
                pass


def test_path_traversal_prevention():
    """Test that path traversal attempts are blocked"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ephemeral_path = temp_path / "ephemeral"
        ephemeral_path.mkdir()

        # Try path traversal
        traversal_path = ephemeral_path / ".." / "sensitive.txt"

        permissions = {
            "read": [str(temp_path)],
            "write": [str(ephemeral_path)],
            "create": [str(ephemeral_path)],
        }

        with pytest.raises(FileAccessViolationError):
            with restricted_open(
                str(traversal_path),
                "w",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            ):
                pass


def test_validate_path():
    """Test path validation logic"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ephemeral_path = temp_path / "ephemeral"
        ephemeral_path.mkdir()

        # Valid paths
        test_file = temp_path / "file.txt"
        test_file.touch()

        subdir_file = temp_path / "subdir" / "file.txt"
        subdir_file.parent.mkdir()
        subdir_file.touch()

        ephemeral_file = ephemeral_path / "file.txt"
        ephemeral_file.touch()

        permissions = {
            "read": [str(temp_path)],
            "write": [str(ephemeral_path)],
            "create": [str(ephemeral_path)],
        }

        # Test valid paths
        assert (
            validate_path(
                test_file,
                mode="r",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            )
            == test_file.resolve()
        )
        assert (
            validate_path(
                subdir_file,
                mode="r",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            )
            == subdir_file.resolve()
        )

        # Test invalid paths
        with pytest.raises(FileAccessViolationError):
            validate_path(
                temp_path.parent / "file.txt",
                mode="r",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            )

        with pytest.raises(FileAccessViolationError):
            validate_path(
                "/etc/passwd",
                mode="r",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            )

        with pytest.raises(FileAccessViolationError):
            validate_path(
                temp_path / ".." / "file.txt",
                mode="r",
                permissions=permissions,
                ephemeral_dir=str(ephemeral_path),
            )


def test_safe_open():
    """Test safe_open wrapper function"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ephemeral_path = temp_path / "ephemeral"
        ephemeral_path.mkdir()

        # Create test file
        test_file = temp_path / "test.txt"
        with open(test_file, "w") as f:
            f.write("test content")

        permissions = {
            "read": [str(temp_path)],
            "write": [str(ephemeral_path)],
            "create": [str(ephemeral_path)],
        }

        # Get safe_open function
        safe_open = get_safe_open(
            ephemeral_dir=str(ephemeral_path), permissions=permissions
        )

        # Test reading
        with safe_open(str(test_file), "r") as f:
            content = f.read()
            assert "test content" in content

        # Test writing
        write_file = ephemeral_path / "write.txt"
        with safe_open(str(write_file), "w") as f:
            f.write("new content")

        # Verify write worked
        with open(write_file) as f:
            assert "new content" in f.read()


def test_extract_paths_absolute():
    task = "Read data from @/path/to/file.txt and @/var/log/app.log"
    paths = extract_paths(task)
    assert len(paths) == 2
    assert any(p.endswith("file.txt") for p in paths)
    assert any(p.endswith("app.log") for p in paths)


def test_extract_paths_relative():
    task = "Process @./data/input.csv and @../backup/data.bak"
    paths = extract_paths(task)
    assert len(paths) == 2
    assert any(p.endswith("input.csv") for p in paths)
    assert any(p.endswith("data.bak") for p in paths)


def test_extract_paths_with_extensions():
    task = "Read @config.yaml and write @output.json"
    paths = extract_paths(task)
    assert len(paths) == 2
    assert any(p.endswith("config.yaml") for p in paths)
    assert any(p.endswith("output.json") for p in paths)


def test_extract_paths_directories():
    task = "Create files in @/tmp/output/ and @backup/"
    paths = extract_paths(task)
    assert len(paths) == 2
    assert any(p.endswith("output") for p in paths)
    assert any(p.endswith("backup") for p in paths)


def test_extract_paths_mixed():
    task = """
    1. Read from @/data/input.txt
    2. Process @./temp/data.csv
    3. Save to @output/
    4. Backup in @/var/backup/data.bak
    """
    paths = extract_paths(task)
    assert len(paths) == 4
    assert any(p.endswith("input.txt") for p in paths)
    assert any(p.endswith("data.csv") for p in paths)
    assert any(p.endswith("output") for p in paths)
    assert any(p.endswith("data.bak") for p in paths)


def test_extract_paths_no_matches():
    task = "Process the data without any file paths"
    paths = extract_paths(task)
    assert len(paths) == 0


def test_extract_paths_duplicates():
    task = "Read @/data/file.txt and process @/data/file.txt again"
    paths = extract_paths(task)
    assert len(paths) == 1
    assert paths[0].endswith("file.txt")


def test_extract_paths_invalid():
    task = "Process @.txt and @/invalid/*/path"
    paths = extract_paths(task)
    assert len(paths) == 0  # Should skip invalid paths


def test_extract_paths_with_spaces():
    task = """Process @/path/to/my\\ file.txt and @/documents/project\\ notes/data.csv and @path/with\\ spaces/file.txt"""
    paths = extract_paths(task)
    assert len(paths) == 3
    assert any(p.endswith("my file.txt") for p in paths)
    assert any(p.endswith("project notes/data.csv") for p in paths)
    assert any(p.endswith("with spaces/file.txt") for p in paths)


def test_extract_paths_with_escapes():
    task = """Process @/path\\ with\\ spaces/file\\ name.txt and @/path\\\\with\\\\backslashes/file.txt"""
    paths = extract_paths(task)
    assert len(paths) == 2
    assert any(p.endswith("path with spaces/file name.txt") for p in paths)
    assert any(p.endswith("path\\with\\backslashes/file.txt") for p in paths)


def test_validate_path_with_permissions():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files and directories
        allowed_dir = temp_path / "allowed"
        ephemeral_dir = temp_path / "ephemeral"
        test_file = allowed_dir / "test.txt"
        ephemeral_file = ephemeral_dir / "eph.txt"

        allowed_dir.mkdir()
        ephemeral_dir.mkdir()
        test_file.touch()
        ephemeral_file.touch()

        # Test read permissions
        permissions = {
            "read": [str(allowed_dir)],
            "write": [str(ephemeral_dir)],
            "create": [str(ephemeral_dir)],
        }

        # Test successful read from allowed path
        assert (
            validate_path(test_file, mode="r", permissions=permissions)
            == test_file.resolve()
        )

        # Test read from non-allowed path
        with pytest.raises(FileAccessViolationError):
            validate_path(
                temp_path / "other" / "file.txt", mode="r", permissions=permissions
            )

        # Test write permissions
        # Write should only be allowed in ephemeral directory
        with pytest.raises(FileAccessViolationError):
            validate_path(
                test_file,
                mode="w",
                ephemeral_dir=str(ephemeral_dir),
                permissions=permissions,
            )

        # Test write in ephemeral directory
        assert (
            validate_path(
                ephemeral_file,
                mode="w",
                ephemeral_dir=str(ephemeral_dir),
                permissions=permissions,
            )
            == ephemeral_file.resolve()
        )

        # Test create permissions
        new_file = ephemeral_dir / "new.txt"
        assert (
            validate_path(
                new_file,
                mode="w",
                ephemeral_dir=str(ephemeral_dir),
                permissions=permissions,
            )
            == new_file.resolve()
        )

        # Test default policy
        other_file = temp_path / "other.txt"
        other_file.touch()

        # With deny policy (default)
        with pytest.raises(FileAccessViolationError):
            validate_path(other_file, mode="r", permissions=permissions)

        # With allow policy
        assert (
            validate_path(
                other_file, mode="r", permissions=permissions, default_policy="allow"
            )
            == other_file.resolve()
        )


def test_restricted_open_with_permissions():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files and directories
        allowed_dir = temp_path / "allowed"
        ephemeral_dir = temp_path / "ephemeral"
        test_file = allowed_dir / "test.txt"
        ephemeral_file = ephemeral_dir / "eph.txt"

        allowed_dir.mkdir()
        ephemeral_dir.mkdir()
        test_file.write_text("test content")
        ephemeral_file.write_text("ephemeral content")

        permissions = {
            "read": [str(allowed_dir)],
            "write": [str(ephemeral_dir)],
            "create": [str(ephemeral_dir)],
        }

        # Test reading from allowed path
        with restricted_open(str(test_file), "r", permissions=permissions) as f:
            assert f.read() == "test content"

        # Test reading from non-allowed path
        with pytest.raises(FileAccessViolationError):
            restricted_open(str(temp_path / "other.txt"), "r", permissions=permissions)

        # Test writing to ephemeral directory
        with restricted_open(
            str(ephemeral_file),
            "w",
            ephemeral_dir=str(ephemeral_dir),
            permissions=permissions,
        ) as f:
            f.write("new content")
        assert ephemeral_file.read_text() == "new content"

        # Test writing outside ephemeral directory
        with pytest.raises(FileAccessViolationError):
            restricted_open(
                str(test_file),
                "w",
                ephemeral_dir=str(ephemeral_dir),
                permissions=permissions,
            )

        # Test creating new file in ephemeral directory
        new_file = ephemeral_dir / "new.txt"
        with restricted_open(
            str(new_file),
            "w",
            ephemeral_dir=str(ephemeral_dir),
            permissions=permissions,
        ) as f:
            f.write("new file")
        assert new_file.read_text() == "new file"


def test_get_safe_open_with_permissions():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files and directories
        allowed_dir = temp_path / "allowed"
        ephemeral_dir = temp_path / "ephemeral"
        test_file = allowed_dir / "test.txt"
        ephemeral_file = ephemeral_dir / "eph.txt"

        allowed_dir.mkdir()
        ephemeral_dir.mkdir()
        test_file.write_text("test content")
        ephemeral_file.write_text("ephemeral content")

        permissions = {
            "read": [str(allowed_dir)],
            "write": [str(ephemeral_dir)],
            "create": [str(ephemeral_dir)],
        }

        safe_open = get_safe_open(
            ephemeral_dir=str(ephemeral_dir), permissions=permissions
        )

        # Test reading from allowed path
        with safe_open(str(test_file), "r") as f:
            assert f.read() == "test content"

        # Test reading from non-allowed path
        with pytest.raises(FileAccessViolationError):
            safe_open(str(temp_path / "other.txt"), "r")

        # Test writing to ephemeral directory
        with safe_open(str(ephemeral_file), "w") as f:
            f.write("new content")
        assert ephemeral_file.read_text() == "new content"

        # Test writing outside ephemeral directory
        with pytest.raises(FileAccessViolationError):
            safe_open(str(test_file), "w")

        # Test creating new file in ephemeral directory
        new_file = ephemeral_dir / "new.txt"
        with safe_open(str(new_file), "w") as f:
            f.write("new file")
        assert new_file.read_text() == "new file"


def test_edge_cases():
    """Test edge cases for file access validation"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"

        permissions = {
            "read": [str(temp_path)],
            "write": [str(temp_path)],
            "create": [str(temp_path)],
        }

        # Test non-existent file
        with pytest.raises(FileAccessViolationError):
            validate_path(test_file, mode="r", permissions=permissions)

        # Test invalid mode
        with pytest.raises(FileAccessViolationError):
            restricted_open(str(test_file), "x", permissions=permissions)

        # Test write without ephemeral dir
        with pytest.raises(FileAccessViolationError):
            restricted_open(str(test_file), "w", permissions=permissions)

        # Test missing permissions
        with pytest.raises(ValueError, match="The permissions parameter is required"):
            validate_path(str(test_file), mode="r")

        # Test non-existent file with read permissions
        with pytest.raises(FileAccessViolationError):
            restricted_open(
                str(temp_path / "nonexistent.txt"), "r", permissions=permissions
            )


def test_get_allowed_paths(tmp_path):
    # Test path validation in get_allowed_paths
    config_permissions = [
        {
            "path": str(tmp_path),
            "access": {"read": True, "write": True},
            "recursive": True,
        }
    ]

    cli_permissions = {
        "read": ["/nonexistent/path"],
        "write": [str(tmp_path)],
        "create": ["/another/nonexistent/path"],
    }

    permissions = get_allowed_paths(config_permissions, cli_permissions)
    assert str(tmp_path) in permissions["write"]
    assert "/nonexistent/path" not in permissions["read"]
    assert "/another/nonexistent/path" not in permissions["create"]


def test_restricted_open_mode_validation():
    """Test validation of file open modes"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"
        test_file.touch()

        permissions = {
            "read": [str(temp_path)],
            "write": [str(temp_path)],
            "create": [str(temp_path)],
        }

        # Test valid modes
        for mode in ["r", "w", "a"]:
            restricted_open(
                str(test_file),
                mode,
                permissions=permissions,
                ephemeral_dir=str(temp_path),
            )

        # Test invalid modes
        for mode in ["x", "rb", "wb", "ab", "r+b", "w+b"]:
            with pytest.raises(FileAccessViolationError, match="Invalid mode"):
                restricted_open(
                    str(test_file),
                    mode,
                    permissions=permissions,
                    ephemeral_dir=str(temp_path),
                )


def test_prevent_modifying_existing_files(tmp_path):
    """Test that existing files cannot be modified without explicit write permission"""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    # Create a test file
    test_file = test_dir / "test.txt"
    test_file.write_text("original content")

    permissions = {
        "read": [str(test_dir)],
        "create": [str(test_dir)],  # Only allow creating new files
    }

    # Try to modify the file without write permission
    with pytest.raises(FileAccessViolationError) as exc:
        validate_path(str(test_file), mode="w", permissions=permissions)
    assert "Writing to" in str(exc.value)

    # Reading should still work
    assert validate_path(str(test_file), mode="r", permissions=permissions)

    # Creating new files should work
    new_file = test_dir / "new.txt"
    assert validate_path(str(new_file), mode="w", permissions=permissions)


def test_parent_directory_permissions(tmp_path):
    """Test that parent directory permissions are checked for new files"""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    permissions = {
        "read": [str(test_dir)],
        "write": [str(test_dir)],
        "create": [str(test_dir)],
    }

    # Try to create a file in a non-existent subdirectory
    non_existent = test_dir / "non_existent" / "test.txt"
    with pytest.raises(FileAccessViolationError) as exc:
        validate_path(str(non_existent), mode="w", permissions=permissions)
    assert "Parent directory" in str(exc.value)

    # Create the subdirectory
    non_existent.parent.mkdir()

    # Now creating the file should work
    assert validate_path(str(non_existent), mode="w", permissions=permissions)


def test_ephemeral_directory_restrictions():
    """Test that ephemeral directory enforces file modification restrictions"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file in the ephemeral directory
        test_file = Path(temp_dir) / "existing.txt"
        test_file.write_text("existing content")

        permissions = {
            "read": [str(temp_dir)],
            "create": [str(temp_dir)],  # Only allow creating new files
        }

        # Try to modify the file in ephemeral directory
        with pytest.raises(FileAccessViolationError) as exc:
            with restricted_open(
                str(test_file), "w", permissions=permissions, ephemeral_dir=temp_dir
            ):
                pass
        assert "Cannot modify existing file in ephemeral directory" in str(exc.value)

        # Should still be able to create new files
        new_file = Path(temp_dir) / "new.txt"
        with restricted_open(
            str(new_file), "w", permissions=permissions, ephemeral_dir=temp_dir
        ):
            pass


def test_mixed_permissions():
    """Test interaction between different permission types"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories with different permissions
        write_dir = Path(temp_dir) / "write_allowed"
        create_only_dir = Path(temp_dir) / "create_only"
        write_dir.mkdir()
        create_only_dir.mkdir()

        # Create test files
        write_file = write_dir / "existing.txt"
        create_only_file = create_only_dir / "existing.txt"
        write_file.write_text("existing content")
        create_only_file.write_text("existing content")

        # Set up mixed permissions
        permissions = {
            "read": [temp_dir],
            "write": [str(write_dir)],
            "create_only": [str(create_only_dir)],
        }

        # Should be able to modify file in write_dir
        with restricted_open(
            str(write_file), "w", permissions=permissions, ephemeral_dir=temp_dir
        ):
            pass

        # Should NOT be able to modify file in create_only_dir
        with pytest.raises(FileAccessViolationError) as exc:
            with restricted_open(
                str(create_only_file),
                "w",
                permissions=permissions,
                ephemeral_dir=temp_dir,
            ):
                pass
        assert "Directory is creation-only" in str(exc.value)

        # Should be able to create new files in both directories
        new_write_file = write_dir / "new.txt"
        new_create_only_file = create_only_dir / "new.txt"

        with restricted_open(
            str(new_write_file), "w", permissions=permissions, ephemeral_dir=temp_dir
        ):
            pass
        with restricted_open(
            str(new_create_only_file),
            "w",
            permissions=permissions,
            ephemeral_dir=temp_dir,
        ):
            pass
