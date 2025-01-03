"""Plan validation functionality."""
from typing import Any, Dict, List

from evolia.models import PlanStep, SystemTool, SystemToolValidation
from evolia.utils.exceptions import ValidationError


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a plan against schema and requirements.

    Args:
        plan: The plan to validate

    Returns:
        Dict[str, Any]: The validated plan

    Raises:
        ValidationError: If validation fails
    """
    # TODO: Implement plan validation
    return plan


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
    validation_errors = []
    matches_interface = True

    # Get the tool interface
    tool = system_tools.get(step.tool)
    if not tool:
        # Special case for built-in tools
        if step.tool in {"generate_code", "execute_code"}:
            return SystemToolValidation(matches_interface=True, validation_errors=[])
        validation_errors.append(f"Unknown tool: {step.tool}")
        return SystemToolValidation(
            matches_interface=False, validation_errors=validation_errors
        )

    # Validate inputs
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
