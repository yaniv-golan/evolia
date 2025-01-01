# Evolia

![License](https://img.shields.io/github/license/yaniv-golan/evolia)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![GitHub Issues](https://img.shields.io/github/issues/yaniv-golan/evolia)
![GitHub Stars](https://img.shields.io/github/stars/yaniv-golan/evolia?style=social)

**Evolia**: AI-powered task orchestration that evolves with your workflow.

Evolia is an experimental task automation system designed to execute user-defined tasks securely and efficiently. It is designed to be reasonably safe and efficient, but use at your own risk.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Contact](#contact)

## Features

- **Schema-Driven Function Interfaces:** Ensures consistency and clarity between the planner and code generator.
- **Enhanced Tool Metadata:** Detailed metadata for each system tool, including dependencies and constraints.
- **File/Folder Access Control:** Automatic and user-specified permissions to secure file operations.
- **Subprocess Policy Management:** Four levels of control over subprocess executions to maintain system integrity.
- **External Libraries Handling:** Detects missing libraries, notifies users, and allows controlled addition of dependencies.
- **Network Access Management:** Facilitates secure internet access with logging, rate limiting, and domain whitelisting.
- **Comprehensive Logging:** Detailed logs for monitoring, auditing, and troubleshooting.
- **Robust Error Handling:** Graceful handling of validation failures and security violations.

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yaniv-golan/evolia.git
   cd evolia
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Configuration**

   Copy the example configuration file and modify as needed.

   ```bash
   cp config.example.yaml config.yaml
   ```

## Configuration

Evolia is highly configurable via the `config.yaml` file and command-line arguments.

### `config.yaml` Structure

```yaml
# config.yaml

# Code Generation Settings
max_syntax_lint_retries: 3
max_runtime_retries: 2

# Library Management
allowed_modules:
  - requests
  - pandas
  - numpy
  - boto3
  - slack_sdk

# Security Settings
security:
  subprocess_policy:
    level: "default"  # Options: default, prompt, system_tool, always
  file_access:
    permissions:
      - path: "/home/user/documents/"
        access:
          read: true
          write: false
          create: true
      - path: "/var/logs/"
        access:
          read: true
          write: false
          create: false

# Network Settings
network:
  whitelist_domains:
    - "api.example.com"
    - "data.example.org"
  rate_limits:
    default:
      requests_per_minute: 60
    api_endpoints:
      - domain: "api.example.com"
        requests_per_minute: 100

# Logging Settings
logging:
  level: "INFO"
  file: "output.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  network_logging: true
  security_logging: true
```

### Command-Line Arguments

**Task Arguments:**

- `--task`: Define the task to execute (required)
- `--keep-artifacts`: Keep temporary artifacts for debugging
- `--no-promotion`: Disable prompting for tool promotion

**Security Settings:**

- `--allow-read <path>`: Grant read access to specified paths
- `--allow-write <path>`: Grant write access to specified paths
- `--allow-create <path>`: Grant create access to specified paths
- `--default-policy {allow,deny}`: Default policy for file access
- `--subprocess-policy {default,prompt,system_tool,always}`: Set subprocess execution policy

**Library Management:**

- `--allow-lib <library>`: Add external libraries to the allowed modules list (format: name[==version])
- `--block-lib <library>`: Block a library at runtime
- `--check-deps`: Check dependencies of allowed libraries
- `--prompt-missing`: Prompt for missing libraries
- `--no-runtime-libs`: Disable runtime library additions

**Network Settings:**

- `--whitelist-domain <domain>`: Add domain to network whitelist
- `--rate-limit <domain> <limit>`: Set rate limit for domain
- `--disable-network-logging`: Disable network request logging

**Logging Settings:**

- `--verbose, -v`: Enable verbose logging output
- `--log-file <path>`: Path to log file (default: output.log)
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set logging level
- `--disable-security-logging`: Disable security event logging

**Example Usage:**

```bash
python -m evolia --task "Process data" \
  --allow-read "/path/to/file" \
  --allow-create "/path/to/folder" \
  --allow-lib boto3 \
  --whitelist-domain "api.example.com" \
  --verbose
```

## Usage

1. **Quick Start (Hello World)**

   Let's start with a simple example:

   ```bash
   python -m evolia --task "Print 'Hello, World!'"
   ```

   This will:
   - Generate a Python function that prints "Hello, World!"
   - Save the output to a file in the run artifacts directory
   - Show you the result in the terminal

2. **Define a Task**

   Tasks are defined using natural language descriptions. Evolia will parse these tasks and generate execution plans accordingly.

3. **Execute a Task**

   ```bash
   python -m evolia --task "Clean the CSV file at /data/input.csv and upload it to AWS S3 bucket 'my-bucket'."
   ```

4. **View Logs**

   All activities are logged in `output.log` for monitoring and auditing purposes.

   ```bash
   tail -f output.log
   ```

## Documentation

Comprehensive documentation is available in the `docs/` directory.

- [User Guide](docs/User_Guide.md)
- [Developer Guide](docs/Developer_Guide.md)
- [API Reference](docs/API_Reference.md)

## Contributing

We welcome contributions from the community! Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## Security

Your security is important to us. Please review our [Security Policies](SECURITY.md) to understand how we handle security-related issues.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or support, please contact [your.email@example.com](mailto:your.email@example.com).

---

*This project is maintained by [Your Name](https://github.com/yaniv-golan).*
