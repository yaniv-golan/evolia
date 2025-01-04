# Evolia: An Experimental CLI Framework for Task Automation via Evolving Code Generation

## Overview

Evolia is an **experimental, command-line driven** framework that explores the automation of complex tasks through the use of LLMs that guide an evolutionary process of code generation and execution. The core idea is simple: a user provides a task description in natural language, and Evolia attempts to generate a solution by creating and executing Python code. This project is a personal exploration into the potential of using LLMs to create self-improving, adaptive systems.

## Core Workflow and the Concept of Evolving Tools

Evolia's approach centers around the concept of evolving code. It works as follows:

1. **Task Definition:** The user provides a natural language description of the task they want to automate via command-line arguments.

2. **Plan Generation:** An LLM analyzes the task and generates a step-by-step execution plan.

3. **Tool Selection and Generation:** Each step in the plan requires a tool to perform a specific action. Evolia uses two types of tools:

    * **System Tools:** These are pre-existing, validated Python functions that serve as building blocks for common operations. Evolia includes a standard library of system tools, and users can define their own. Proven generated tools can also be promoted to system tools.

    * **Candidate Tools:** If no suitable system tool exists, Evolia's LLM-powered Code Generator creates new Python code. This code is initially considered a "candidate" tool.

4. **Code Validation:** All generated code (candidate tools) undergoes rigorous validation to ensure it is syntactically correct, adheres to predefined constraints, and meets security standards.

5. **Plan Execution:** Evolia's Executor executes the plan step-by-step, either running system tools or executing candidate tools within a secure, sandboxed environment.

6. **Evolution through Promotion:** As candidate tools are used successfully across multiple tasks, they are tracked by the Candidate Manager. Tools that prove to be both robust and frequently used can be **promoted** to become system tools. This evolutionary process allows Evolia to gradually build a library of reliable tools, improving its ability to handle future tasks more efficiently.

7. **Iterative Refinement:** If a step fails during execution, or if the overall task is not completed successfully, Evolia can use feedback from the validation and execution stages to refine the plan or regenerate the code. This iterative approach allows the system to adapt and learn from its mistakes.

## Key Components

* **Plan Generator:** An LLM-based component that generates a step-by-step plan to fulfill a given task.
* **Code Generator:** Leverages an LLM to produce Python code for candidate tools.
* **Code Fixer:** Attempts to automatically repair errors detected in generated code.
* **Executor:** Manages the secure execution of system tools and candidate tools.
* **Library Manager:** Controls the use of external Python libraries, enforcing security and version constraints.
* **Candidate Manager:** Tracks and manages generated code snippets (candidate tools), monitoring their usage and success rate.
* **Tool Promoter:** Enables the promotion of validated and frequently used candidate tools to system tool status.

## Experimental Nature and Personal Motivation

Evolia is an **experimental** project, created primarily as a personal exploration into the potential of LLM-driven, evolutionary code generation. The goal is to understand how far such a system can go in automating complex tasks and adapting to new challenges. As such, the framework is constantly evolving, and its capabilities are likely to change significantly over time.

The project's experimental nature means that:

* **It is not intended for production use (yet).** The focus is on exploration and learning, not building a fully polished and stable product.
* **There may be limitations and unexpected behaviors.** The system is based on cutting-edge AI techniques, which are inherently probabilistic and can produce unpredictable results.
* **The architecture and design are subject to change.** As new insights are gained, the framework may be significantly restructured or redesigned.

## Getting Started

Evolia is a **command-line application** written in Python.

**Installation:**

Since Evolia is not yet on PyPI, you need to clone it directly from GitHub:

```bash
git clone https://github.com/yaniv-golan/evolia.git
cd evolia
pip install -e .
```

**Configuration:**

Evolia uses a YAML configuration file (`config.yaml`) to control various aspects of its behavior, such as allowed libraries, file access permissions, and API keys. An example configuration file is included in the repository.

**Example Usage:**

To execute a task using Evolia, you would run it from the command line like this:

```bash
evolia --task "Read data from 'input.csv', calculate the average of the 'value' column, and write the result to 'output.txt'" --allow-read ./ --allow-write ./ --allow-create ./ --default-policy allow
```

**Note:** this command allows read, write, and create in the current directory. These permissions are only for the sake of the example.

This command tells Evolia to:

1. Use the task description provided after `--task`.
2. Use the specified read/write/create permissions.

You can also use the `--config` flag to specify a different configuration file. See `python evolia --help` for more details on available command-line options.

## Contributing

While Evolia is primarily a personal project, contributions and feedback are welcome. If you are interested in exploring the code or suggesting improvements, please feel free to open issues or pull requests on the GitHub repository.

## License

Evolia is released under the **MIT License**. See the `LICENSE` file for more details.

For more examples and detailed documentation, see the docs/ directory.
