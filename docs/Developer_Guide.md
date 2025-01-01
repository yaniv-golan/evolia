# Evolia Developer Guide

Welcome to the Evolia Developer Guide! This document provides in-depth information for developers looking to contribute to or extend Evolia.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Core Components](#core-components)
  - [Core Module](#core-module)
  - [Models Module](#models-module)
  - [Security Module](#security-module)
  - [Network Module](#network-module)
  - [Validation Module](#validation-module)
  - [Utils Module](#utils-module)
  - [Integrations Module](#integrations-module)
  - [Configuration Module](#configuration-module)
- [Extending Functionality](#extending-functionality)
  - [Adding New Tools](#adding-new-tools)
  - [Defining Function Interfaces](#defining-function-interfaces)
  - [Implementing New Security Policies](#implementing-new-security-policies)
- [Testing](#testing)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
- [Contribution Workflow](#contribution-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

Evolia follows a modular architecture, allowing each component to function independently while interacting seamlessly with others. The primary components include:

- **Planner:** Interprets user tasks and generates execution plans.
- **Executor2:** Carries out the execution plans with controlled access and security measures, providing enhanced validation and error handling.
- **Tool Loader:** Dynamically loads and validates system tools based on metadata.
- **Security Module:** Enforces security policies across file access, subprocess executions, and network interactions.
- **Configuration Manager:** Manages system configurations via `config.yaml` and CLI arguments.
- **Logging Module:** Handles comprehensive logging for monitoring and auditing.

## Project Structure

```
evolia/
├── evolia.py
├── config.example.yaml
├── config.yaml
├── system_tools.json
├── requirements.txt
├── LICENSE
├── README.md
├── SECURITY.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── src/
│   └── evolia/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── config_manager.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── executor2.py
│       │   ├── planner.py
│       │   ├── tool_loader.py
│       │   ├── code_generator.py
│       │   ├── code_fixer.py
│       │   ├── function_generator.py
│       │   ├── interface_verification.py
│       │   ├── restricted_execution.py
│       │   └── candidate_manager.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   └── models.py
│       ├── security/
│       │   ├── __init__.py
│       │   ├── security.py
│       │   └── file_access.py
│       ├── validation/
│       │   ├── __init__.py
│       │   └── code_validation.py
│       ├── network/
│       │   ├── __init__.py
│       │   ├── network_logging.py
│       │   └── rate_limiter.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── exceptions.py
│       │   └── logger.py
│       └── integrations/
           ├── __init__.py
           └── external_apis.py
└── docs/
    ├── User_Guide.md
    ├── Developer_Guide.md
    ├── API_Reference.md
```

## Setting Up the Development Environment

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yaniv-golan/evolia.git
   cd evolia
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the System**

   Copy the example configuration file and modify it as needed.

   ```bash
   cp config.example.yaml config.yaml
   ```

## Core Components

### Core Module

**Role:** Contains the main execution and planning components of the system.

**Key Components:**

- **Executor2 (`core/executor2.py`):** Executes generated code with security measures and validation
- **Planner (`core/planner.py`):** Generates execution plans from tasks
- **Code Generator (`core/code_generator.py`):** Generates code from function interfaces
- **Code Fixer (`core/code_fixer.py`):** Fixes syntax and runtime errors in generated code
- **Function Generator (`core/function_generator.py`):** Creates function implementations
- **Interface Verification (`core/interface_verification.py`):** Validates tool interfaces
- **Restricted Execution (`core/restricted_execution.py`):** Handles secure code execution
- **Candidate Manager (`core/candidate_manager.py`):** Manages code generation candidates

### Models Module

**Role:** Defines data structures and schemas used throughout the system.

**Key Components:**

- **Schemas (`models/schemas.py`):** Core data structure definitions
- **Models (`models/models.py`):** Pydantic models for data validation

### Security Module

**Role:** Enforces security policies across various operational aspects.

**Key Components:**

- **Security Visitor (`security/security.py`):** AST-based security enforcement
- **File Access (`security/file_access.py`):** Controls file system operations

### Network Module

**Role:** Manages network operations and monitoring.

**Key Components:**

- **Network Logging (`network/network_logging.py`):** Logs network activity
- **Rate Limiter (`network/rate_limiter.py`):** Controls API request rates

### Validation Module

**Role:** Validates code and ensures security compliance.

**Key Components:**

- **Code Validation (`validation/code_validation.py`):** Validates generated code

### Utils Module

**Role:** Provides utility functions and error handling.

**Key Components:**

- **Exceptions (`utils/exceptions.py`):** Custom exception classes
- **Logger (`utils/logger.py`):** Logging configuration and utilities

### Integrations Module

**Role:** Handles external service integrations.

**Key Components:**

- **External APIs (`integrations/external_apis.py`):** Manages external API interactions

### Configuration Module

**Role:** Manages system configuration and settings.

**Key Components:**

- **Config Manager (`config/config_manager.py`):** Handles configuration loading and validation

## Extending Functionality

### Adding New Tools

1. **Implement the Tool**

   Create a new Python module in the `tools/system/` directory.

   ```python
   # tools/system/new_tool.py

   def new_function(input_param: str) -> str:
       # Implementation here
       return "output"
   ```

2. **Define Tool Metadata**

   Add a new entry in `system_tools.json`.

   ```json
   {
     "name": "new_tool",
     "filepath": "tools/system/new_tool.py",
     "description": "Description of the new tool.",
     "inputs": ["input_param"],
     "outputs": ["output"],
     "dependencies": ["library>=version"],
     "constraints": ["Any constraints"],
     "version": "1.0.0",
     "interface": {
       "function_name": "new_function",
       "parameters": [
         {
           "name": "input_param",
           "type": "str",
           "description": "Description of the input parameter."
         }
       ],
       "return_type": "str",
       "description": "Description of what the function does.",
       "examples": [
         "new_function('input')  # Returns 'output'"
       ],
       "constraints": [
         "Any constraints or conditions."
       ]
     }
   }
   ```

3. **Update Pydantic Models**

   Ensure that the new tool adheres to the defined Pydantic schemas for validation.

4. **Run Tests**

   Add unit tests to verify the functionality and integration of the new tool.

### Defining Function Interfaces

Function interfaces are crucial for maintaining consistency between the Planner and Code Generator.

1. **Create Function Interface Schema**

   Define the function interface using Pydantic models in `schemas.py`.

   ```python
   # evolia/schemas.py

   from pydantic import BaseModel, Field
   from typing import List, Optional

   class FunctionParameter(BaseModel):
       name: str
       type: str
       description: Optional[str] = Field("", description="Description of the parameter's purpose and constraints.")

   class FunctionInterface(BaseModel):
       function_name: str
       parameters: List[FunctionParameter]
       return_type: str
       description: Optional[str] = Field("", description="Description of what the function does.")
       examples: Optional[List[str]] = Field([], description="Example usages of the function.")
       constraints: Optional[List[str]] = Field([], description="Any constraints or special conditions for the function.")
   ```

2. **Integrate with Tools**

   Ensure each tool's metadata includes a detailed `interface` field as demonstrated in the Tool Loader section.

### Implementing New Security Policies

1. **Define the Policy**

   Determine the scope and rules of the new security policy.

2. **Update Configuration**

   Add the new policy settings to `config.yaml`.

3. **Implement Enforcement Mechanism**

   Modify the `SecurityVisitor` class or relevant modules to enforce the new policy.

4. **Test the Policy**

   Create test cases to validate that the policy works as intended without introducing vulnerabilities.

## Testing

### Unit Tests

- **Location:** `tests/unit/`
- **Framework:** `pytest`
- **Purpose:** Test individual components and functions for correctness.

**Example Test:**

```python
# tests/unit/test_file_access.py

import pytest
from evolia.file_access import validate_path

def test_validate_path_allowed_read():
    allowed_paths = ["/home/user/documents/"]
    assert validate_path("/home/user/documents/report.csv", allowed_paths, "/tmp") == True

def test_validate_path_disallowed_write():
    allowed_paths = ["/home/user/documents/"]
    assert validate_path("/home/user/documents/report.csv", allowed_paths, "/tmp", create_new=False) == False
```

### Integration Tests

- **Location:** `tests/integration/`
- **Framework:** `pytest`
- **Purpose:** Test the interaction between multiple components to ensure cohesive functionality.

**Example Test:**

```python
# tests/integration/test_executor.py

import pytest
from evolia.core.executor2 import Executor2 as Executor
from evolia.config import ConfigModel

def test_execute_tool():
    config = ConfigModel()
    executor = Executor(config)
    step = PlanStep(description="Clean CSV at /data/input.csv and upload to S3.")
    result = executor.execute_code(step, 1)
    assert "s3_url" in result
```

## Contribution Workflow

1. **Fork the Repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yaniv-golan/evolia.git
   cd evolia
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Implement Your Changes**

   Develop your feature or fix within the new branch.

5. **Run Tests**

   Ensure all tests pass.

   ```bash
   pytest
   ```

6. **Commit Your Changes**

   ```bash
   git add .
   git commit -m "Add feature: YourFeatureName"
   ```

7. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeatureName
   ```

8. **Open a Pull Request**

   Navigate to your fork on GitHub and open a pull request to the main repository.

## Code Style Guidelines

- **Language:** Python 3.8+
- **Style Guide:** [PEP 8](https://pep8.org/)
- **Type Hints:** Use type annotations for all function signatures.
- **Documentation:** Ensure all modules, classes, and functions have descriptive docstrings.

## Troubleshooting

### Common Issues

1. **Dependency Conflicts**

   **Issue:** Installation fails due to conflicting library versions.

   **Solution:** Check the `requirements.txt` for specified versions and ensure compatibility. Consider using a virtual environment.

2. **Configuration Errors**

   **Issue:** Incorrect settings in `config.yaml` lead to unexpected behavior.

   **Solution:** Validate the YAML syntax and refer to the [Configuration](../docs/User_Guide.md#configuration) section for correct formats.

3. **Permission Denied**

   **Issue:** Accessing certain files or directories is denied.

   **Solution:** Ensure that the paths are correctly specified in `file_access.permissions` with appropriate access rights.

### Seeking Help

If you encounter issues not covered in this guide, please open an issue in the [GitHub Issues](https://github.com/yaniv-golan/evolia/issues) section or contact the maintainers at [your.email@example.com](mailto:your.email@example.com).

---

*Happy coding! 🚀*
