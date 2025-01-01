"""Tests for library management functionality"""
import pytest
from evolia.core.library_management import (
    detect_missing_libraries,
    validate_library_version,
    get_library_dependencies
)

@pytest.fixture
def library_config():
    """Test configuration for library management"""
    return {
        'allowed_modules': {
            'requests': '2.25.0',
            'pandas': '1.2.0',
            'numpy': '1.19.0',
            'json': None,
            'os': None,
            'pathlib': None
        },
        'library_management': {
            'check_dependencies': True,
            'prompt_for_missing': True,
            'allow_runtime_additions': False,
            'blocked_libraries': ['subprocess', 'socket'],
            'runtime_overrides': {
                'allowed': [],
                'blocked': []
            }
        }
    }

def test_detect_missing_libraries():
    """Test detection of missing libraries"""
    # Test with existing libraries
    assert detect_missing_libraries(['os', 'sys', 'json']) == []
    
    # Test with missing libraries
    missing = detect_missing_libraries(['nonexistent_lib', 'another_missing_lib'])
    assert 'nonexistent_lib' in missing
    assert 'another_missing_lib' in missing

def test_validate_library_version():
    """Test library version validation"""
    # Test with installed library
    is_valid, version = validate_library_version('pip')
    assert is_valid
    assert version is not None
    
    # Test with version requirement met
    is_valid, version = validate_library_version('pip', '0.1.0')
    assert is_valid
    
    # Test with version requirement not met
    is_valid, version = validate_library_version('pip', '99.99.99')
    assert not is_valid
    
    # Test with non-existent library
    is_valid, version = validate_library_version('nonexistent_lib')
    assert not is_valid
    assert version is None

def test_get_library_dependencies():
    """Test getting library dependencies"""
    # Test with a library that has dependencies
    deps = get_library_dependencies('requests')
    assert len(deps) > 0
    assert 'urllib3' in deps
    
    # Test with a library that doesn't exist
    deps = get_library_dependencies('nonexistent_lib')
    assert len(deps) == 0

def test_library_manager_init(library_config):
    """Test LibraryManager initialization"""
    manager = LibraryManager(library_config)
    assert 'requests' in manager.allowed_libraries
    assert 'subprocess' not in manager.allowed_libraries
    assert manager.library_versions['requests'] == '2.25.0'

def test_library_manager_validate_imports(library_config):
    """Test validation of imports"""
    manager = LibraryManager(library_config)
    
    # Test allowed imports
    errors = manager.validate_imports({'json', 'os', 'pathlib'})
    assert len(errors) == 0
    
    # Test disallowed imports
    errors = manager.validate_imports({'subprocess', 'socket'})
    assert len(errors) == 2
    assert any('subprocess' in error for error in errors)
    assert any('socket' in error for error in errors)
    
    # Test version requirements
    errors = manager.validate_imports({'requests'})
    assert len(errors) == 0 or any('version' in error for error in errors)

def test_library_manager_check_dependencies(library_config):
    """Test checking library dependencies"""
    manager = LibraryManager(library_config)
    
    # Test with allowed library
    disallowed = manager.check_dependencies({'json'})
    assert len(disallowed) == 0
    
    # Test with library that has dependencies
    disallowed = manager.check_dependencies({'requests'})
    assert len(disallowed) > 0  # requests has several dependencies

def test_library_manager_add_allowed_library(library_config):
    """Test adding allowed libraries"""
    manager = LibraryManager(library_config)
    
    # Add new library
    manager.add_allowed_library('new_lib', '1.0.0')
    assert 'new_lib' in manager.allowed_libraries
    assert manager.library_versions['new_lib'] == '1.0.0'
    
    # Add without version
    manager.add_allowed_library('another_lib')
    assert 'another_lib' in manager.allowed_libraries
    assert 'another_lib' not in manager.library_versions 