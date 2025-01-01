# Evolia API Reference

This document provides detailed information about the APIs and internal interfaces of Evolia. It is intended for developers who wish to extend or integrate with Evolia.

## Table of Contents

- [Core Module](#core-module)
  - [Executor2](#executor2)
  - [Planner](#planner)
  - [Code Generator](#code-generator)
  - [Code Fixer](#code-fixer)
  - [Function Generator](#function-generator)
  - [Interface Verification](#interface-verification)
  - [Restricted Execution](#restricted-execution)
  - [Candidate Manager](#candidate-manager)
- [Models](#models)
  - [Schemas](#schemas)
  - [Data Models](#data-models)
- [Security](#security)
  - [Security Visitor](#security-visitor)
  - [File Access](#file-access)
- [Network](#network)
  - [Network Logging](#network-logging)
  - [Rate Limiter](#rate-limiter)
- [Validation](#validation)
  - [Code Validation](#code-validation)
- [Utils](#utils)
  - [Exceptions](#exceptions)
  - [Logger](#logger)
- [Integrations](#integrations)
  - [External APIs](#external-apis)
- [Configuration](#configuration)
  - [Config Manager](#config-manager)

---

## Core Module

### Executor2

The main execution engine for running generated code with security measures.

```python
class Executor2:
    def execute_code(self, step: PlanStep, step_num: int) -> Dict[str, Any]:
        """
        Executes a single step of the plan.

        Parameters:
            step (PlanStep): The plan step to execute.
            step_num (int): The step number in the plan.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - Direct return values from the executed code's main() function
                - Any files created in the step directory
                - If the code sets a 'result' key, it will be displayed as the final result
        """

    def execute_tool(self, tool: Tool, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a specific tool with provided inputs.

        Parameters:
            tool (Tool): The tool to execute.
            inputs (Dict[str, Any]): Inputs required by the tool.

        Returns:
            Dict[str, Any]: Outputs produced by the tool.
        """
```

### Code Generator

Generates code from function interfaces and requirements.

```python
class CodeGenerator:
    def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResponse:
        """
        Generates code based on the provided request.

        Parameters:
            request (CodeGenerationRequest): Contains function interface and requirements.

        Returns:
            CodeGenerationResponse: Generated code and metadata.
        """
```

### Code Fixer

Fixes syntax and runtime errors in generated code.

```python
class CodeFixer:
    def fix_syntax_error(self, code: str, error: SyntaxError) -> str:
        """
        Attempts to fix syntax errors in the code.

        Parameters:
            code (str): The code containing syntax errors.
            error (SyntaxError): The syntax error details.

        Returns:
            str: Fixed code.
        """

    def fix_runtime_error(self, code: str, error: Exception) -> str:
        """
        Attempts to fix runtime errors in the code.

        Parameters:
            code (str): The code containing runtime errors.
            error (Exception): The runtime error details.

        Returns:
            str: Fixed code.
        """
```

## Models

### Schemas

Core data structure definitions.

```python
CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "language": {"type": "string"},
        "imports": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "object"}
    },
    "required": ["code", "language"]
}
```

### Data Models

Pydantic models for data validation.

```python
class PlanStep(BaseModel):
    """Represents a single step in an execution plan."""
    description: str
    code: str
    inputs: Dict[str, Any]
    outputs: Dict[str, OutputDefinition]

class Tool(BaseModel):
    """Represents a system tool with metadata."""
    name: str
    filepath: str
    description: str
    inputs: List[str]
    outputs: List[str]
    dependencies: Optional[List[str]] = []
    constraints: Optional[List[str]] = []
    version: Optional[str] = "1.0.0"
    interface: FunctionInterface
```

## Security

### Security Visitor

AST-based security enforcement.

```python
class SecurityVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import):
        """
        Visits import statements to enforce library whitelisting.

        Parameters:
            node (ast.Import): The import node to inspect.
        """

    def visit_Call(self, node: ast.Call):
        """
        Visits function calls to enforce subprocess policies.

        Parameters:
            node (ast.Call): The call node to inspect.
        """
```

### File Access

Controls file system operations.

```python
def validate_path(path: str, access_type: str) -> bool:
    """
    Validates if a path can be accessed with the specified access type.

    Parameters:
        path (str): The path to validate.
        access_type (str): The type of access (read/write/create).

    Returns:
        bool: Whether access is allowed.
    """

def get_safe_open(path: str, mode: str) -> IO:
    """
    Gets a safe file handle with the specified mode.

    Parameters:
        path (str): The path to open.
        mode (str): The file mode.

    Returns:
        IO: A file handle.
    """
```

## Network

### Network Logging

Logs network activity.

```python
def log_network_request(url: str, method: str, headers: Dict[str, str]):
    """
    Logs a network request.

    Parameters:
        url (str): The request URL.
        method (str): The HTTP method.
        headers (Dict[str, str]): The request headers.
    """
```

### Rate Limiter

Controls API request rates.

```python
class RateLimiter:
    def check_rate_limit(self, key: str) -> bool:
        """
        Checks if a request should be rate limited.

        Parameters:
            key (str): The rate limit key.

        Returns:
            bool: Whether the request is allowed.
        """
```
