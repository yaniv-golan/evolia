# Test Structure

This directory contains all tests for the Evolia project. The tests are organized as follows:

## Directory Structure

```
tests/
├── integration/          # Integration tests
│   ├── api/             # API endpoint tests
│   ├── core/            # Core functionality tests
│   │   ├── code_generation/  # Code generation integration tests
│   │   │   ├── test_code_generation_output.py
│   │   │   ├── test_code_generator.py
│   │   │   ├── test_code_validation.py
│   │   │   └── test_live_api.py
│   │   └── execution/   # Code execution integration tests
│   └── utils/           # Utility function tests
├── unit/                # Unit tests
│   ├── core/            # Core component tests
│   │   ├── code_generation/  # Code generation unit tests
│   │   │   ├── test_code_generator.py
│   │   │   ├── test_code_fixer.py
│   │   │   └── test_function_generator.py
│   │   ├── execution/   # Execution unit tests
│   │   │   └── test_executor.py
│   │   ├── test_evolia.py  # Core functionality tests
│   │   └── test_promotion.py  # Tool promotion tests
│   ├── models/          # Data model tests
│   ├── security/        # Security feature tests
│   ├── utils/           # Utility function tests
│   │   └── test_exceptions.py  # Exception handling tests
│   └── validation/      # Input validation tests
│       ├── code/        # Code validation tests
│       │   └── test_schema_validation.py
│       ├── interface/   # Interface validation tests
│       │   └── test_interface_verification.py
│       └── plan/        # Plan validation tests
│           ├── test_plan_interface_validation.py
│           └── test_validation.py
├── data/                # Test data
│   ├── fixtures/        # Test fixtures
│   │   ├── api/        # API test fixtures
│   │   ├── core/       # Core functionality fixtures
│   │   └── utils/      # Utility test fixtures
│   └── samples/         # Sample files for testing
│       ├── code/       # Sample code files
│       ├── configs/    # Configuration files
│       │   └── system_tools.json  # System configuration
│       └── tools/      # Tool definition files
├── conftest.py          # Shared pytest fixtures
└── requirements-test.txt # Test dependencies
```

## Test Categories

### Unit Tests

- Located in `unit/`
- Test individual components in isolation
- Mock external dependencies
- Fast execution
- Organized by feature area:
  - `core/`: Core functionality tests
    - `code_generation/`: Code generation components
    - `execution/`: Code execution components
    - Core system tests (evolia.py, promotion.py)
  - `models/`: Data model validation
  - `security/`: Security features and restrictions
  - `utils/`: Utility functions and exceptions
  - `validation/`: Input validation
    - `code/`: Schema and code validation
    - `interface/`: Interface verification
    - `plan/`: Plan validation and verification

### Integration Tests

- Located in `integration/`
- Test multiple components working together
- Minimal mocking
- May include external dependencies
- Organized by feature area:
  - `api/`: API endpoint tests
  - `core/`: Core functionality integration
    - `code_generation/`: Code generation integration
    - `execution/`: Code execution integration
  - `utils/`: Utility integration tests

### Test Data

- Located in `data/`
- Organized by purpose and feature area:
  - `fixtures/`: Test fixtures
    - `api/`: API test fixtures
    - `core/`: Core functionality fixtures
    - `utils/`: Utility test fixtures
  - `samples/`: Sample files
    - `code/`: Sample code files for testing
    - `configs/`: Configuration files
    - `tools/`: Tool definition files

## Testing Guidelines

### Unit vs Integration Tests

1. Unit Tests:
   - Test a single function/method/class
   - Mock all external dependencies
   - Focus on edge cases and error handling
   - Should be fast and independent

2. Integration Tests:
   - Test interaction between components
   - Use real dependencies where possible
   - Focus on common use cases
   - Can be slower and more complex

### Test Coverage

1. Code Generation:
   - Unit tests for individual components
   - Integration tests for end-to-end generation
   - Live API tests in separate file
   - Validation tests for all constraints

2. Code Execution:
   - Unit tests for executor components
   - Integration tests for full execution
   - Security restriction tests
   - Resource management tests

3. API and Utils:
   - Unit tests for utility functions
   - Integration tests for API endpoints
   - Error handling coverage
   - Edge case coverage

### Test Data Management

1. Fixtures:
   - Use for repeatable test setup
   - Organize by feature area
   - Keep focused and minimal
   - Document purpose and usage

2. Sample Files:
   - Use realistic examples
   - Maintain versioning
   - Document dependencies
   - Keep size manageable

## Conventions

1. File Naming:
   - Test files should be named `test_*.py`
   - Match the name of the module being tested
   - Example: `test_code_generator.py` tests `code_generator.py`

2. Test Functions:
   - Name test functions descriptively
   - Follow pattern: `test_<what>_<expected_outcome>`
   - Example: `test_invalid_syntax_output`

3. Fixtures:
   - Place shared fixtures in `conftest.py`
   - Module-specific fixtures in test files
   - Use clear, descriptive names

4. Assertions:
   - Use pytest assertions
   - Include descriptive messages
   - Be specific about what is being tested

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py

# Run tests with coverage
pytest --cov=evolia

# Run tests in parallel
pytest -n auto

# Run specific test category
pytest tests/unit/  # Run all unit tests
pytest tests/integration/  # Run all integration tests

# Run specific feature tests
pytest tests/unit/core/code_generation/  # Run code generation unit tests
pytest tests/integration/core/execution/  # Run execution integration tests
```

## Adding New Tests

1. Identify the appropriate category (unit/integration)
2. Place in the correct subdirectory based on feature area
3. Follow naming conventions
4. Include necessary fixtures
5. Add to test coverage report

## Best Practices

1. Keep tests focused and atomic
2. Use appropriate fixtures
3. Clean up test resources
4. Document complex test scenarios
5. Maintain test data separately
6. Group related tests by feature area
7. Use appropriate mocking strategy
8. Include both positive and negative test cases
9. Separate unit and integration concerns
10. Keep test files focused and manageable
