# Evolia

A Python code generation and execution framework that uses LLMs to generate and fix code.

## Code Generation

Evolia uses a single function-based code generation approach through the `FunctionGenerator` class. This specialized generator:

1. Generates single functions with strict interface requirements
2. Includes chain-of-thought (COT) reasoning for GPT-4 models only
3. Validates generated code against security and syntax requirements
4. Supports automatic code fixing through `CodeFixer`

### Chain of Thought Reasoning

The code generation process includes conditional chain-of-thought reasoning:

- For GPT-4 models (e.g., gpt-4o-2024-08-06): Full COT reasoning is included
- For other models: COT section is omitted for more concise output

The COT templates are defined in `prompts.py` and are only injected for GPT-4 models:

- `FUNCTION_COT_TEMPLATE`: Used for function generation
- `FIX_COT_TEMPLATE`: Used for code fixing

### Code Fixing

The `CodeFixer` class provides automatic code fixing capabilities:

1. Uses the same COT approach as function generation (only for GPT-4)
2. Tracks fix history to avoid repeated attempts
3. Maintains original function signatures
4. Validates fixes against security and syntax requirements

## Configuration

The main configuration is in `config.yaml` (or `default.yaml`):

1. Model settings (e.g., which GPT model to use)
2. Security policies
3. Code generation templates
4. Allowed modules and built-ins

## Usage

Basic usage:

```python
from evolia import Evolia

# Initialize with task
evolia = Evolia(task="add 88 and 99")

# Run the task
result = evolia.run()
```

For more examples and detailed documentation, see the docs/ directory.
