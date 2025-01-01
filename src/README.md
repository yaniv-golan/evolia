# Source Code Structure

The `evolia` package is organized into several modules:

## Core (`core/`)

- `evolia.py` - Main Evolia class and functionality
- `executor2.py` - Enhanced code execution and plan execution with improved validation and error handling
- `promotion.py` - Code promotion and deployment
- `library_management.py` - Python library management

## Utils (`utils/`)

- `logger.py` - Logging functionality
- `exceptions.py` - Custom exceptions

## Integrations (`integrations/`)

- `openai_structured.py` - OpenAI API integration with structured output

## Models (`models/`)

- `models.py` - Data models and validation

## Network (`network/`)

- `network_logging.py` - Network request logging
- `network_rate_limiter.py` - Rate limiting for network requests

## Security (`security/`)

- `file_access.py` - File access control and validation
- `security.py` - Security validation and checks

## Validation (`validation/`)

- `code_validation.py` - Code validation and safety checks

## Configuration (`config/`)

- `config.yaml` - Main configuration file
- `default.yaml` - Default configuration template
