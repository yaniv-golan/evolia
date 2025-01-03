"""Main Evolia module."""

import os
import sys
import json
import logging
import argparse
import yaml
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

from ..models import (
    Plan,
    PlanStep,
    CodeGenerationRequest,
    GeneratedCode,
    Parameter,
    CodeGenerationResponse,
    CodeResponse,
    ExecutionRequest,
    ExecutionResponse,
    TestCase,
    TestResults,
    ValidationResults,
    SystemTool,
    FunctionInterface,
    InterfaceValidation,
    SystemToolValidation,
    GenerateCodeValidation,
    ExecuteCodeValidation,
    StepValidationBase,
    OutputDefinition,
)
from ..integrations.openai_structured import call_openai_structured
from ..security.file_access import get_safe_open, FileAccessViolationError
from .executor2 import Executor2
from .interface_verification import verify_interface
from .promotion import ToolPromoter, PromotionError
from .candidate_manager import CandidateManager
from ..utils.logger import (
    setup_logger,
    code_generation_context,
    validation_context,
    execution_context,
)
from ..validation.code_validation import validate_python_code
from ..security.security import validate_code_security
from ..utils.exceptions import (
    CodeGenerationError,
    CodeValidationError,
    CodeExecutionError,
    SecurityViolationError,
    PlanValidationError,
    PlanExecutionError,
    PlanGenerationError,
    ExecutorError,
)

logger = logging.getLogger("evolia")

PLAN_GENERATION_PROMPT = """You are a plan generator that creates strictly linear execution plans.
Each step must have:
- name: A descriptive name for the step
- tool: Either "generate_code", "execute_code", or the name of a system tool
- inputs: Dictionary of input parameters
- outputs: Dictionary of step outputs where each output specifies its type
- allowed_read_paths: List of paths the step can read from
- allowed_write_paths: List of paths the step can write to
- allowed_create_paths: List of paths the step can create files in
- default_policy: Default policy for file access ('allow' or 'deny')
- interface_validation: Object containing validation results

You must validate each step's interface against its tool's requirements.
For system tools, check the interface field in the tool's metadata.
For generate_code and execute_code, follow the standard interface rules.
Include validation results in each step's interface_validation field.

Interface Validation Rules:
1. For system tools:
   - Function name must match exactly
   - Parameter names must match exactly
   - Parameter types must match exactly
   - Return type must match exactly
   - All required parameters must be provided
   - No extra parameters allowed
2. For generate_code:
   - Function name must be a valid Python identifier
   - Parameter names must be valid Python identifiers
   - Parameter types must be from the allowed set
   - Return type must be from the allowed set
   - Outputs must include "code_file" with type "str"
3. For execute_code:
   - Must reference a valid Python file in run_artifacts/tmp directory
   - Must provide all required function parameters
   - Output types must match the function's return type
   Example: For task "calculate area of rectangle with width 5 and height 3":
   {
     "name": "Calculate rectangle area",
     "tool": "execute_code",
     "inputs": {
       "script_file": "run_artifacts/tmp/calculate_area.py",
       "width": 5,
       "height": 3
     },
     "outputs": {"result": {"type": "float"}}
   }

Output Rules:
1. Each output must specify its type:
   outputs: {
     "output_name": {"type": "output_type"}
   }
2. Types must match the tool's interface:
   - For generate_code: code_file must be type "str"
   - For execute_code: type must match function's return type
   - For system tools: type must match tool's interface
3. Output types must be from the allowed set

File Access Rules:
1. Each step must specify its file access permissions:
   - allowed_read_paths: List of paths the step can read from
   - allowed_write_paths: List of paths the step can write to
   - allowed_create_paths: List of paths the step can create files in
   - default_policy: 'allow' or 'deny' for paths not explicitly listed
2. Paths should be relative to the workspace root
3. Use the most restrictive permissions needed for each step
4. Default policy should be 'deny' unless broader access is required"""

# JSON Schema for OpenAI structured output
PLAN_SCHEMA = {
    "type": "object",
    "description": "A structured plan for executing a task as a series of steps",
    "properties": {
        "steps": {
            "type": "array",
            "description": "Ordered sequence of steps to execute",
            "items": {
                "type": "object",
                "description": "A single execution step in the plan",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Descriptive name for the step",
                    },
                    "tool": {
                        "type": "string",
                        "description": "The tool to use for this step (generate_code, execute_code, or a system tool name)",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Input parameters for the step",
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": "Name of the function to generate",
                                "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                            },
                            "parameters": {
                                "type": "array",
                                "description": "List of function parameters",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Parameter name",
                                            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                                        },
                                        "type": {
                                            "type": "string",
                                            "description": "Parameter type",
                                            "enum": [
                                                "Tuple",
                                                "Union",
                                                "bool",
                                                "float",
                                                "Any",
                                                "Dict",
                                                "str",
                                                "int",
                                                "set",
                                                "dict",
                                                "list",
                                                "tuple",
                                                "Set",
                                                "Optional",
                                                "List",
                                            ],
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Description of the parameter",
                                        },
                                    },
                                    "required": ["name", "type"],
                                    "additionalProperties": False,
                                },
                            },
                            "return_type": {
                                "type": "string",
                                "description": "Function return type",
                                "enum": [
                                    "Tuple",
                                    "Union",
                                    "bool",
                                    "float",
                                    "Any",
                                    "Dict",
                                    "str",
                                    "int",
                                    "set",
                                    "dict",
                                    "list",
                                    "tuple",
                                    "Set",
                                    "Optional",
                                    "List",
                                ],
                            },
                            "description": {
                                "type": "string",
                                "description": "Clear description of the function's purpose",
                            },
                            "examples": {
                                "type": "array",
                                "description": "Example usages of the function",
                                "items": {"type": "string"},
                            },
                            "constraints": {
                                "type": "array",
                                "description": "List of constraints or limitations",
                                "items": {"type": "string"},
                                "default": ["no_globals", "pure_function"],
                            },
                            "script_file": {
                                "type": "string",
                                "description": "Path to the Python script to execute",
                                "pattern": "^run_artifacts/.*\\.py$",
                            },
                        },
                        "additionalProperties": True,
                    },
                    "outputs": {
                        "type": "object",
                        "description": "Step outputs with their types",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "Tuple",
                                        "Union",
                                        "bool",
                                        "float",
                                        "Any",
                                        "Dict",
                                        "str",
                                        "int",
                                        "set",
                                        "dict",
                                        "list",
                                        "tuple",
                                        "Set",
                                        "Optional",
                                        "List",
                                    ],
                                    "description": "The type of this output",
                                }
                            },
                            "required": ["type"],
                            "additionalProperties": False,
                        },
                        "minProperties": 1,
                    },
                    "allowed_read_paths": {
                        "type": "array",
                        "description": "Paths that this step is allowed to read from",
                        "items": {"type": "string"},
                    },
                    "allowed_write_paths": {
                        "type": "array",
                        "description": "Paths that this step is allowed to write to",
                        "items": {"type": "string"},
                    },
                    "allowed_create_paths": {
                        "type": "array",
                        "description": "Paths that this step is allowed to create files in",
                        "items": {"type": "string"},
                    },
                    "default_policy": {
                        "type": "string",
                        "description": "Default policy for file access: 'allow' or 'deny'",
                        "enum": ["allow", "deny"],
                        "default": "deny",
                    },
                    "interface_validation": {
                        "type": "object",
                        "description": "Validation results for function interface",
                        "properties": {
                            "matches_interface": {
                                "type": "boolean",
                                "description": "Whether the step matches its tool's interface",
                            },
                            "validation_errors": {
                                "type": "array",
                                "description": "List of interface validation errors",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["matches_interface"],
                    },
                },
                "required": [
                    "name",
                    "tool",
                    "inputs",
                    "outputs",
                    "allowed_read_paths",
                    "allowed_write_paths",
                    "allowed_create_paths",
                    "default_policy",
                ],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
        "artifacts_dir": {
            "type": "string",
            "description": "Directory for storing generated artifacts",
            "default": "run_artifacts",
        },
    },
    "required": ["steps", "artifacts_dir"],
    "additionalProperties": False,
}


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    logger = logging.getLogger("evolia")
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    logger.debug(f"Loading configuration from {config_path}")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logger.debug(
                "Configuration loaded successfully",
                extra={
                    "payload": {
                        "config_keys": list(config.keys()),
                        "allowed_modules": config.get("allowed_modules", []),
                    }
                },
            )
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}", exc_info=True)
        raise


def load_system_tools() -> Dict[str, SystemTool]:
    """Load available system tools from system_tools.json"""
    logger = logging.getLogger("evolia")
    tools_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "system_tools.json"
    )
    logger.debug(f"Loading system tools from {tools_path}")
    try:
        with open(tools_path) as f:
            tools_data = json.load(f)
            tools = {}
            for tool_data in tools_data:
                # Convert interface parameters to Parameter objects
                interface = tool_data.get("interface", {})
                parameters = [
                    Parameter(
                        name=p["name"],
                        type=p["type"],
                        description=p.get("description", ""),
                    )
                    for p in interface.get("parameters", [])
                ]

                # Convert outputs to OutputDefinition objects
                outputs = {}
                if "return_type" in interface:
                    outputs["result"] = OutputDefinition(type=interface["return_type"])

                # Create SystemTool object
                tool = SystemTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    parameters=parameters,
                    outputs=outputs,
                    permissions=tool_data.get("permissions"),
                    filepath=tool_data.get("filepath"),
                )
                tools[tool.name] = tool

            logger.debug(
                "System tools loaded successfully",
                extra={
                    "payload": {
                        "tool_count": len(tools),
                        "tool_names": list(tools.keys()),
                        "tool_paths": [t.filepath for t in tools.values()],
                    }
                },
            )
            return tools
    except Exception as e:
        logger.error(f"Failed to load system tools: {str(e)}", exc_info=True)
        raise


def validate_step_interface(
    step: PlanStep, system_tools: Dict[str, SystemTool]
) -> StepValidationBase:
    """Validate that a step's interface matches its tool's requirements."""
    if step.tool == "generate_code":
        return GenerateCodeValidation.from_step(step)
    elif step.tool == "execute_code":
        return ExecuteCodeValidation.from_step(step)
    elif step.tool in system_tools:
        return SystemToolValidation.from_step(step, system_tools[step.tool])
    else:
        validation = StepValidationBase()
        validation.matches_interface = False
        validation.validation_errors = [f"Unknown tool: {step.tool}"]
        return validation


def generate_plan(
    task: str, system_tools: Dict[str, SystemTool], config: Dict[str, Any], args: Any
) -> Plan:
    """Generate an execution plan for a task.

    Args:
        task: Task description
        system_tools: Dictionary of available system tools
        config: Configuration dictionary
        args: Command line arguments (can be dict or argparse.Namespace)

    Returns:
        Plan: Generated execution plan

    Raises:
        PlanGenerationError: If plan generation fails
    """
    logger = logging.getLogger("evolia")
    logger.info("Generating execution plan")

    try:
        # Convert system tools to dictionaries for serialization
        serializable_tools = {
            name: tool.to_dict() for name, tool in system_tools.items()
        }

        # Handle both dict and argparse.Namespace objects
        allow_read = args["allow_read"] if isinstance(args, dict) else args.allow_read
        allow_write = (
            args["allow_write"] if isinstance(args, dict) else args.allow_write
        )
        allow_create = (
            args["allow_create"] if isinstance(args, dict) else args.allow_create
        )
        default_policy = (
            args["default_policy"] if isinstance(args, dict) else args.default_policy
        )

        # Call OpenAI to generate plan
        response = call_openai_structured(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=config["openai"]["model"],
            temperature=config["openai"]["temperature"],
            json_schema=PLAN_SCHEMA,
            user_prompt=f"""Create a plan for the following task: "{task}"

Available system tools:
{json.dumps(serializable_tools, indent=2)}

File Access Permissions:
- Allowed read paths: {allow_read}
- Allowed write paths: {allow_write}
- Allowed create paths: {allow_create}
- Default policy: {default_policy}

Plan Structure:
1. For simple file I/O tasks (read/process/write), use a SINGLE step that:
   - Reads from the input file
   - Processes the data
   - Writes to the output file
2. Only break into multiple steps if there are complex dependencies or transformations
3. Each step should use either generate_code, execute_code, or a system tool
4. Steps should be ordered logically with clear dependencies
5. When using system tools, strictly follow their defined function interfaces
6. Validate that each step's inputs and outputs match the tool's interface
7. Use the provided file access permissions for each step

IMPORTANT: 
1. For generate_code steps:
   - outputs must include "code_file" with a .py file path in run_artifacts directory
   - inputs must include:
     - function_name: A descriptive name for the function (must be valid Python identifier)
     - parameters: List of parameters with name (valid Python identifier) and type
     - return_type: Expected return type from allowed types
     - description: Clear description of the function's purpose
     - examples: List of usage examples (optional)
     - constraints: List of any constraints or limitations (optional)
   - Include any necessary imports in the generated code
   - For file I/O, implement the actual file operations in the function body
   - DO NOT use __name__ == '__main__' blocks
2. For execute_code steps:
   - inputs must include:
     - script_file: Path to the Python file to execute (must be in run_artifacts)
     - Any additional inputs needed by the function
   - outputs must include expected return values
3. For system tool steps:
   - inputs must EXACTLY match the tool's interface parameters:
     - Parameter names must match exactly
     - Parameter types must match exactly
     - All required parameters must be provided
   - outputs must match the tool's defined return type exactly
   - respect any constraints defined in the tool's metadata
   - validate interface compatibility before execution
4. All file paths must be within the run_artifacts directory
5. Each step must have all required fields (name, tool, inputs, outputs, allowed_read_paths)
6. Each step must include interface_validation with:
   - matches_interface: boolean indicating if the step matches its tool's interface
   - validation_errors: list of any validation errors found
7. Each step must include file access permissions:
   - allowed_read_paths: List of paths the step can read from
   - allowed_write_paths: List of paths the step can write to
   - allowed_create_paths: List of paths the step can create files in
   - default_policy: Default policy for file access ('allow' or 'deny')""",
            system_prompt=PLAN_GENERATION_PROMPT,
        )

        # Convert dictionary steps to PlanStep objects
        steps = []
        for step_dict in response["steps"]:
            step = PlanStep.from_dict(step_dict)

            # Validate step interface
            step.interface_validation = validate_step_interface(step, system_tools)

            steps.append(step)

        # Create plan
        plan = Plan(steps=steps, artifacts_dir=response["artifacts_dir"])

        logger.info("Plan generated successfully")
        return plan

    except Exception as e:
        logger.error(f"Failed to generate plan: {str(e)}")
        raise PlanGenerationError(f"Failed to generate plan: {str(e)}")


def validate_plan(
    plan: Plan, system_tools: Dict[str, SystemTool], config: Dict[str, Any]
) -> List[str]:
    """Validate an execution plan.

    Args:
        plan: Plan to validate
        system_tools: Dictionary of available system tools
        config: Configuration dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    with validation_context("plan"):
        # Validate each step
        for i, step in enumerate(plan.steps):
            # Check tool exists
            if (
                step.tool not in ["generate_code", "execute_code"]
                and step.tool not in system_tools
            ):
                errors.append(f"Step {i + 1}: Unknown tool '{step.tool}'")
                continue

            # Get tool interface
            if step.tool in ["generate_code", "execute_code"]:
                interface = None  # Standard interface
            else:
                tool = system_tools[step.tool]
                interface = tool.interface

            # Validate interface
            interface_errors = (
                verify_interface(step.inputs, interface) if interface else []
            )
            if interface_errors:
                errors.extend(f"Step {i + 1}: {error}" for error in interface_errors)

            # Validate file access paths
            if not isinstance(step.allowed_read_paths, list):
                errors.append(f"Step {i + 1}: allowed_read_paths must be a list")
            if not isinstance(step.allowed_write_paths, list):
                errors.append(f"Step {i + 1}: allowed_write_paths must be a list")
            if not isinstance(step.allowed_create_paths, list):
                errors.append(f"Step {i + 1}: allowed_create_paths must be a list")

            # Validate default policy
            if step.default_policy not in ["allow", "deny"]:
                errors.append(f"Step {i + 1}: default_policy must be 'allow' or 'deny'")

    if errors:
        raise PlanValidationError("Plan validation failed", {"errors": errors})

    return errors


def execute_plan(
    plan: Plan, system_tools: Dict[str, SystemTool], config: Dict[str, Any], args: Any
) -> None:
    """Execute a validated plan.

    Args:
        plan: Plan to execute
        system_tools: Dictionary of available system tools
        config: Configuration dictionary
        args: Command line arguments
    """
    executor = Executor2(config)

    with execution_context("plan"):
        try:
            for i, step in enumerate(plan.steps):
                logger.info(f"Executing step {i + 1}: {step.name}")

                # Set up file access for this step
                executor.set_file_access(
                    step.allowed_read_paths,
                    step.allowed_write_paths,
                    step.allowed_create_paths,
                    step.default_policy,
                )

                # Execute the step
                if step.tool == "generate_code":
                    executor.generate_code(step.inputs, step.outputs)
                elif step.tool == "execute_code":
                    executor.execute_code(step.inputs, step.outputs)
                else:
                    # Get and execute system tool
                    tool = system_tools[step.tool]
                    executor.execute_tool(tool, step.inputs, step.outputs)

                logger.info(f"Step {i + 1} completed successfully")

        except (ExecutorError, SecurityViolationError, FileAccessViolationError) as e:
            raise PlanExecutionError(
                f"Failed to execute step {i + 1}: {step.name}",
                step.name,
                {"error": str(e)},
            )


def main():
    """Main entry point for Evolia."""
    # Load environment variables first
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evolia: AI-powered task orchestration that evolves with your workflow."
    )

    # File access control arguments
    file_access_group = parser.add_argument_group("File Access Control")
    file_access_group.add_argument(
        "--allow-read", nargs="+", help="Paths to allow reading from", default=[]
    )
    file_access_group.add_argument(
        "--allow-write", nargs="+", help="Paths to allow writing to", default=[]
    )
    file_access_group.add_argument(
        "--allow-create", nargs="+", help="Paths to allow creating files in", default=[]
    )
    file_access_group.add_argument(
        "--default-policy",
        choices=["allow", "deny"],
        default="deny",
        help="Default policy for file access",
    )

    # ECS-related arguments
    ecs_group = parser.add_argument_group("Ephemeral-Candidate-System")
    ecs_group.add_argument("--tag-candidate", help="Move ephemeral code to candidates")
    ecs_group.add_argument(
        "--promote-candidate", help="Promote a candidate to system tool"
    )
    ecs_group.add_argument(
        "--auto-promote",
        action="store_true",
        help="When tagging, mark candidate for auto-promotion",
    )
    ecs_group.add_argument("--description", help="Description for promoted tool")

    # Library management arguments
    lib_group = parser.add_argument_group("Library Management")
    lib_group.add_argument(
        "--allow-lib",
        action="append",
        help="Allow a library at runtime (format: name[==version])",
    )
    lib_group.add_argument(
        "--block-lib", action="append", help="Block a library at runtime"
    )
    lib_group.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies of allowed libraries",
    )
    lib_group.add_argument(
        "--prompt-missing", action="store_true", help="Prompt for missing libraries"
    )
    lib_group.add_argument(
        "--no-runtime-libs",
        action="store_true",
        help="Disable runtime library additions",
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )
    log_group.add_argument(
        "--log-file", type=str, help="Path to log file (default: output.log)"
    )

    # Task arguments
    parser.add_argument("--task", required=True, help="High-level task description")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temporary artifacts for debugging",
    )

    args = parser.parse_args()

    # Set up logging
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logger(log_file=log_file, verbose=args.verbose)

    # Handle ECS commands first
    if args.tag_candidate:
        try:
            manager = CandidateManager()
            new_path = manager.move_to_candidates(args.tag_candidate, args.auto_promote)
            logger.info(f"Successfully moved to candidates: {new_path}")
            return 0
        except Exception as e:
            logger.error(f"Error tagging candidate: {str(e)}")
            return 1

    if args.promote_candidate:
        try:
            manager = CandidateManager()
            promoter = ToolPromoter()

            # Get candidate metadata
            candidate = manager.get_candidate_stats(args.promote_candidate)
            if not candidate:
                logger.error(f"Candidate not found: {args.promote_candidate}")
                return 1

            # Get candidate file path
            candidate_path = Path(candidate["filepath"])
            if not candidate_path.exists():
                logger.error(f"Candidate file not found: {candidate_path}")
                return 1

            # Promote to system tool
            new_path = promoter.promote_candidate_to_system(
                str(candidate_path), candidate, description=args.description
            )
            logger.info(f"Successfully promoted to system tool: {new_path}")
            return 0

        except Exception as e:
            logger.error(f"Error promoting candidate: {str(e)}")
            return 1

    # Log startup information
    logger.info("Starting Evolia")
    logger.debug(
        "Environment information",
        extra={
            "payload": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": os.getcwd(),
                "env_vars": {
                    k: "[REDACTED]" if "TOKEN" in k else v
                    for k, v in os.environ.items()
                    if k.startswith(("OPENAI_", "PATH"))
                },
            }
        },
    )

    logger.info(f"Task: {args.task}")
    logger.debug("Command line arguments", extra={"payload": {"args": vars(args)}})

    # Load config and system tools
    config = load_config()
    logger.debug("Configuration loaded")

    system_tools = load_system_tools()
    logger.debug(f"Loaded {len(system_tools)} system tools")

    try:
        # Update file access configuration with CLI arguments
        if not "file_access" in config:
            config["file_access"] = {}

        # Initialize runtime_overrides if not present
        if "runtime_overrides" not in config["file_access"]:
            config["file_access"]["runtime_overrides"] = {
                "read": [],
                "write": [],
                "create": [],
            }

        # Map CLI arguments to runtime overrides
        config["file_access"]["runtime_overrides"]["read"] = [
            os.path.realpath(p) for p in args.allow_read
        ]
        config["file_access"]["runtime_overrides"]["write"] = [
            os.path.realpath(p) for p in args.allow_write
        ]
        config["file_access"]["runtime_overrides"]["create"] = [
            os.path.realpath(p) for p in args.allow_create
        ]

        # Update default policy if specified
        if args.default_policy:
            config["file_access"]["default_policy"] = args.default_policy

        logger.debug(
            "Updated file access configuration",
            extra={"payload": {"file_access": config["file_access"]}},
        )

        # Initialize library management section if not present
        if "library_management" not in config:
            config["library_management"] = {
                "check_dependencies": False,
                "prompt_for_missing": False,
                "allow_runtime_additions": True,
                "runtime_overrides": {"allowed": [], "blocked": []},
            }

        # Update library management settings from CLI
        if args.allow_lib:
            config["library_management"]["runtime_overrides"]["allowed"] = []
            for lib in args.allow_lib:
                if "==" in lib:
                    name, version = lib.split("==")
                    config["library_management"]["runtime_overrides"]["allowed"].append(
                        {"name": name.strip(), "version": version.strip()}
                    )
                else:
                    config["library_management"]["runtime_overrides"]["allowed"].append(
                        {"name": lib.strip()}
                    )

        if args.block_lib:
            config["library_management"]["runtime_overrides"]["blocked"] = [
                lib.strip() for lib in args.block_lib
            ]

        if args.check_deps is not None:
            config["library_management"]["check_dependencies"] = args.check_deps

        if args.prompt_missing is not None:
            config["library_management"]["prompt_for_missing"] = args.prompt_missing

        if args.no_runtime_libs:
            config["library_management"]["allow_runtime_additions"] = False

        # Generate and execute plan
        logger.info("Generating plan...")
        start_time = time.time()
        plan = generate_plan(args.task, system_tools, config, args)
        logger.info("Plan generated successfully")

        # Display the plan in a human-readable format
        print("\nGenerated Plan:")
        print("-------------")
        for i, step in enumerate(plan.steps, 1):
            print(f"\nStep {i}: {step.name}")
            print(f"Tool: {step.tool}")
            print("Inputs:")
            for key, value in step.inputs.items():
                print(f"  - {key}: {value}")
            print("Outputs:")
            for key, value in step.outputs.items():
                print(f"  - {key}: {value}")
            print("File Access Permissions:")
            if step.allowed_read_paths:
                print("  Read paths:")
                for path in step.allowed_read_paths:
                    print(f"    - {path}")
            if step.allowed_write_paths:
                print("  Write paths:")
                for path in step.allowed_write_paths:
                    print(f"    - {path}")
            if step.allowed_create_paths:
                print("  Create paths:")
                for path in step.allowed_create_paths:
                    print(f"    - {path}")
            print(f"  Default policy: {step.default_policy}")
        print("\nExecuting plan...\n")

        logger.debug(
            "Plan generation completed",
            extra={
                "payload": {
                    "generation_time": time.time() - start_time,
                    "plan": plan.model_dump(),
                }
            },
        )

        # Execute plan
        ephemeral_dir = config.get("file_access", {}).get(
            "ephemeral_dir", "run_artifacts"
        )
        executor = Executor2(
            config=config,
            keep_artifacts=args.keep_artifacts,
            ephemeral_dir=ephemeral_dir,
        )
        logger.info("Executing plan...")
        start_time = time.time()
        executor.execute_plan(plan)
        logger.info(
            "Plan execution completed",
            extra={
                "payload": {
                    "execution_time": time.time() - start_time,
                    "generated_files": executor.generated_files,
                }
            },
        )

    except (CodeGenerationError, CodeValidationError) as e:
        logger.error(
            f"Code generation/validation error: {str(e)}",
            extra={
                "payload": {
                    "code": getattr(e, "code", ""),
                    "details": getattr(e, "details", {}),
                    "validation_results": getattr(e, "validation_results", {}),
                }
            },
        )
        exit(1)
    except SecurityViolationError as e:
        logger.error(
            f"Security violation: {str(e)}",
            extra={
                "payload": {
                    "violations": getattr(e, "violations", {}),
                    "code": getattr(e, "code", ""),
                }
            },
        )
        exit(1)
    except (ExecutorError, CodeExecutionError) as e:
        logger.error(
            f"Execution error: {str(e)}",
            extra={
                "payload": {
                    "test_results": getattr(e, "test_results", {}),
                    "execution_error": getattr(e, "execution_error", None),
                }
            },
        )
        exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
