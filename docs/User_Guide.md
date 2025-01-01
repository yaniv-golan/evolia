# Evolia User Guide

Welcome to the Evolia User Guide! This document provides comprehensive instructions on how to effectively use and configure Evolia to automate your tasks securely.

## Table of Contents

- [Getting Started](#getting-started)
- [Defining Tasks](#defining-tasks)
- [Configuration](#configuration)
  - [Configuration File (`config.yaml`)](#configuration-file-configyaml)
  - [Command-Line Arguments](#command-line-arguments)
- [File/Folder Access Control](#filefolder-access-control)
  - [Automatic Access Based on Instructions](#automatic-access-based-on-instructions)
  - [User-Specified Access](#user-specified-access)
- [Subprocess Policy Management](#subprocess-policy-management)
  - [Policy Levels](#policy-levels)
  - [Configuring Subprocess Policies](#configuring-subprocess-policies)
- [Managing External Libraries](#managing-external-libraries)
  - [Detecting Missing Libraries](#detecting-missing-libraries)
  - [Adding Allowed Libraries](#adding-allowed-libraries)
  - [Installing Libraries](#installing-libraries)
- [Network Access Management](#network-access-management)
  - [Logging Network Requests](#logging-network-requests)
  - [Rate Limiting](#rate-limiting)
  - [Domain Whitelisting](#domain-whitelisting)
- [Logging and Monitoring](#logging-and-monitoring)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Getting Started

### Installation

Refer to the [Installation](../README.md#installation) section in the `README.md` for detailed instructions on setting up Evolia.

### Initial Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yaniv-golan/evolia.git
   cd evolia
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the System**

   Copy the example configuration file and customize it as needed.

   ```bash
   cp config.example.yaml config.yaml
   ```

## Defining Tasks

Tasks in Evolia are defined using natural language descriptions. The system parses these descriptions to generate execution plans.

**Example Task:**

```bash
python -m evolia --task "Process the data file"
```

## Configuration

Evolia offers flexible configuration options through the `config.yaml` file and command-line arguments.

### Configuration File (`config.yaml`)

The `config.yaml` file centralizes all configuration settings.

**Example `config.yaml`:**

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

Enhance or override configuration settings using CLI flags.

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

## File/Folder Access Control

Evolia provides robust mechanisms to control file and folder access, ensuring that tasks operate within defined boundaries.

### Automatic Access Based on Instructions

The system automatically grants read access to files and folders mentioned in task descriptions and allows the creation of new files within those directories.

**Implementation Details:**

- **Path Extraction:** Parses task descriptions to identify file paths using regex.
  
  ```python
  import re

  def extract_paths(task_description: str) -> List[str]:
      pattern = r'[/]?[\w\-/\.]+\.[\w]+'
      return re.findall(pattern, task_description)
  ```

- **Permission Enforcement:** Allows reading existing files and creating new ones without modifying or deleting existing content.

### User-Specified Access

Users can explicitly define additional file and folder permissions via the `config.yaml` file or CLI arguments.

**Configuration Example:**

```yaml
file_access:
  permissions:
    - path: "/home/user/documents/"
      access:
        read: true
        write: false
        create: true
```

**CLI Example:**

```bash
python -m evolia --task "Analyze logs" --allow-read "/var/logs/" --allow-create "/var/logs/new/"
```

**Implementation Details:**

- **Merging Permissions:** Combines permissions from both configurations and CLI, with CLI taking precedence.
- **Validation:** Ensures paths and permissions are valid and do not grant excessive access.

## Subprocess Policy Management

Evolia controls subprocess executions through a flexible policy system with four levels of control.

### Policy Levels

1. **Do Not Allow (Default):** Subprocess calls are disallowed.
2. **Allow Only After User Approval Mid-Run:** Requires explicit user consent.
3. **Allow if Invoked by a System Tool (Promoted Tool):** Only trusted tools can invoke subprocesses.
4. **Allow Always:** Subprocess calls are permitted without restrictions.

### Configuring Subprocess Policies

Define the desired policy level in the `config.yaml` file.

**Example:**

```yaml
subprocess_policy:
  level: "prompt"  # Options: default, prompt, system_tool, always
```

**Implementation Details:**

- **Policy Enforcement:** Utilizes the `SecurityVisitor` to monitor and enforce subprocess policies during code execution.
- **User Prompts:** For the `prompt` level, the system will request user approval when a subprocess call is detected.

## Managing External Libraries

Evolia handles external library dependencies meticulously to maintain security and functionality.

### Detecting Missing Libraries

The system analyzes generated code to identify any imports not listed in the `allowed_modules`.

**Implementation Example:**

```python
import ast

def detect_missing_libraries(code: str, allowed_modules: List[str]) -> List[str]:
    tree = ast.parse(code)
    imported = {alias.name.split('.')[0] for alias in tree.body if isinstance(alias, ast.Import) for alias in alias.names}
    imported |= {node.module.split('.')[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    missing = list(imported - set(allowed_modules))
    return missing
```

### Adding Allowed Libraries

Users can add external libraries either via the `config.yaml` file or using CLI flags.

**Configuration Example:**

```yaml
allowed_modules:
  - requests
  - pandas
  - boto3
  - slack_sdk
```

**CLI Example:**

```bash
python -m evolia --task "Upload data" --allow-lib boto3 --allow-lib slack_sdk
```

### Installing Libraries

After adding a library to the allowed list, install it using `pip`.

**Manual Installation:**

```bash
pip install boto3 slack_sdk
```

**Automated Installation (Optional):**

Implement a feature that prompts the user to install missing libraries automatically.

```python
import subprocess
import sys

def install_libraries(libs: List[str]):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    except subprocess.CalledProcessError:
        print("Failed to install libraries. Please install them manually.")
        sys.exit(1)
```

### Verifying Library Installation

Ensure that all allowed libraries are installed and can be imported.

```python
def verify_libraries(libs: List[str]):
    for lib in libs:
        try:
            __import__(lib)
        except ImportError:
            print(f"Library {lib} could not be imported. Please check the installation.")
            raise
```

## Network Access Management

Evolia facilitates controlled internet access with comprehensive security measures.

### Logging Network Requests

All outbound network requests are logged for monitoring and auditing.

**Implementation Example:**

```python
import requests
import logging

logger = logging.getLogger('evolia.network')

class LoggedRequests:
    def __init__(self):
        self.session = requests.Session()

    def get(self, *args, **kwargs):
        logger.info(f"Making GET request to {args[0]}")
        return self.session.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        logger.info(f"Making POST request to {args[0]}")
        return self.session.post(*args, **kwargs)
```

### Rate Limiting

Prevent excessive network usage by limiting the number of requests within a specified period.

**Implementation Example:**

```python
import time
import logging

logger = logging.getLogger('evolia.network')

class RateLimiter:
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period
        self.call_times = []

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            current_time = time.time()
            self.call_times = [t for t in self.call_times if current_time - t < self.period]
            if len(self.call_times) >= self.max_calls:
                logger.warning("Rate limit exceeded for network requests.")
                raise Exception("Rate limit exceeded.")
            self.call_times.append(current_time)
            return func(*args, **kwargs)
        return wrapped
```

### Domain Whitelisting

Optionally restrict network access to specific, trusted domains to enhance security.

**Configuration Example:**

```yaml
network:
  whitelist_domains:
    - "api.example.com"
    - "data.example.org"
```

**Implementation Example:**

```python
from urllib.parse import urlparse
from typing import List

class LoggedRequests:
    def __init__(self, whitelist_domains: List[str] = []):
        self.session = requests.Session()
        self.whitelist_domains = whitelist_domains

    def is_whitelisted(self, url: str) -> bool:
        domain = urlparse(url).netloc
        return domain in self.whitelist_domains

    @rate_limiter
    def get(self, *args, **kwargs):
        url = args[0]
        if self.whitelist_domains and not self.is_whitelisted(url):
            logger.warning(f"GET request to {url} is not whitelisted.")
            raise Exception(f"GET request to {url} is not allowed.")
        logger.info(f"Making GET request to {url}")
        return self.session.get(*args, **kwargs)

    @rate_limiter
    def post(self, *args, **kwargs):
        url = args[0]
        if self.whitelist_domains and not self.is_whitelisted(url):
            logger.warning(f"POST request to {url} is not whitelisted.")
            raise Exception(f"POST request to {url} is not allowed.")
        logger.info(f"Making POST request to {url}")
        return self.session.post(*args, **kwargs)
```

## Logging and Monitoring

Comprehensive logging ensures that all critical operations are recorded for auditing and troubleshooting.

- **General Logs:** Capture standard operations and tool executions.
- **Security Logs:** Document security-related events like permission violations and subprocess calls.
- **Network Logs:** Record all outbound network requests, including URLs and response statuses.

**Log Configuration Example:**

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("evolia.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('evolia')
```

## Troubleshooting

### Missing Libraries

**Issue:** The generated code requires `boto3`, but it's not installed.

**Solution:**

1. Install the required library:

   ```bash
   pip install boto3
   ```

2. Add it to the allowed modules:

   ```bash
   python -m evolia --task "Upload data" --allow-lib boto3
   ```

### Subprocess Call Denied

**Issue:** Subprocess call was denied by policy.

**Solution:**

Update the `config.yaml` to change the `subprocess_policy.level` to `prompt` or `always`, depending on your needs.

```yaml
subprocess_policy:
  level: "prompt"  # Options: default, prompt, system_tool, always
```

### Permission Errors

**Issue:** Access to `/path/to/file` with mode `w` is denied.

**Solution:**

Ensure that the path is included in `file_access.permissions` with `create: true`.

```yaml
file_access:
  permissions:
    - path: "/path/to/"
      access:
        read: true
        write: false
        create: true
```

### Default Policy

If you encounter permission errors when accessing files or directories:

1. **Check File Access Configuration**
   - Verify that the paths are correctly specified in `config.yaml`
   - Ensure the paths exist and your user has appropriate system permissions
   - Check that `file_access.default_policy` is set appropriately:
     - If set to `deny` (default), you must explicitly allow each path
     - If set to `allow`, paths are accessible unless explicitly denied
     - Example: `Permission denied when reading file.txt` might mean you need to add the file's path to allowed_read_paths or change default_policy to "allow"

2. **Subprocess Permission Issues**
   - Verify the subprocess policy level in configuration
   - Check if the command is in the allowed_commands list
   - Consider using the 'prompt' policy for testing

## Best Practices

- **Least Privilege:** Grant only the necessary permissions required for each task.
- **Regular Updates:** Keep all dependencies and Evolia updated to the latest versions.
- **Monitor Logs:** Regularly review logs to detect and respond to any suspicious activities.
- **Secure Configuration:** Carefully configure subprocess policies and network access to match your security requirements.
- **Validate Inputs:** Ensure that all inputs and configurations are validated to prevent injection attacks or path traversal vulnerabilities.

---

*For more detailed information, refer to the [Developer Guide](Developer_Guide.md) and [API Reference](API_Reference.md) in the `docs/` directory.*

## Usage Examples

```bash
# Basic task execution
python -m evolia --task "Process the data file"

# Task with file access and library
python -m evolia --task "Clean CSV and upload to S3" \
  --allow-read "/data/input.csv" \
  --allow-lib boto3

# Task with network access
python -m evolia --task "Fetch data from API" \
  --whitelist-domain "api.example.com" \
  --rate-limit "api.example.com" 100
```
