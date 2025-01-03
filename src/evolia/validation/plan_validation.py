"""Plan validation functionality."""
from typing import Any, Dict, List

from evolia.models import (
    ExecuteCodeValidation,
    Plan,
    PlanStep,
    SystemTool,
    SystemToolValidation,
)
from evolia.utils.exceptions import PlanValidationError, ValidationError


def validate_plan(
    plan: Plan, system_tools: Dict[str, SystemTool], config: Dict[str, Any]
) -> List[str]:
    """Validate a plan against schema and requirements.

    Args:
        plan: The plan to validate
        system_tools: Dictionary of available system tools
        config: Configuration dictionary

    Returns:
        List[str]: List of validation errors. Empty list if validation passes.

    Raises:
        PlanValidationError: If validation fails
    """
    errors = []

    # Validate each step in the plan
    for step in plan.steps:
        # Validate path fields are lists
        if not isinstance(step.allowed_read_paths, list):
            errors.append("allowed_read_paths must be a list")
        if not isinstance(step.allowed_write_paths, list):
            errors.append("allowed_write_paths must be a list")
        if not isinstance(step.allowed_create_paths, list):
            errors.append("allowed_create_paths must be a list")

        # Validate default policy
        if step.default_policy not in ["allow", "deny"]:
            errors.append("default_policy must be 'allow' or 'deny'")

        # Validate the tool interface
        validation_result = validate_step_interface(step, system_tools)
        if not validation_result.matches_interface:
            errors.extend(validation_result.validation_errors)

    if errors:
        raise PlanValidationError("\n".join(errors))

    return []


def validate_step_interface(
    step: PlanStep, system_tools: Dict[str, SystemTool]
) -> SystemToolValidation:
    """Validate a step against its interface.

    Args:
        step: The step to validate
        system_tools: Dictionary of available system tools

    Returns:
        SystemToolValidation: The validation result
    """

    # Get the tool interface
    tool = system_tools.get(step.tool)
    if not tool:
        # Special case for built-in tools
        if step.tool == "generate_code":
            return SystemToolValidation(matches_interface=True, validation_errors=[])
        elif step.tool == "execute_code":
            # Use ExecuteCodeValidation for execute_code steps
            result = ExecuteCodeValidation.from_step(step)
            return result
        else:
            return SystemToolValidation(
                matches_interface=False,
                validation_errors=[f"Unknown tool: {step.tool}"],
            )

    # Validate inputs
    validation_errors = []
    matches_interface = True

    required_inputs = {
        param.name for param in tool.parameters if not getattr(param, "optional", False)
    }
    provided_inputs = set(step.inputs.keys())

    missing_inputs = required_inputs - provided_inputs
    if missing_inputs:
        matches_interface = False
        validation_errors.append(
            f"Missing required inputs: {', '.join(missing_inputs)}"
        )

    # Validate outputs
    for output_name, output_def in step.outputs.items():
        if output_name not in tool.outputs:
            matches_interface = False
            validation_errors.append(f"Unknown output: {output_name}")
        elif output_def.type != tool.outputs[output_name].type:
            matches_interface = False
            validation_errors.append(
                f"Output type mismatch for {output_name}: "
                f"expected {tool.outputs[output_name].type}, got {output_def.type}"
            )

    return SystemToolValidation(
        matches_interface=matches_interface, validation_errors=validation_errors
    )
