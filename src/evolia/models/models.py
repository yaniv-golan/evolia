from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Union, Tuple

# Allowed types for function parameters and return values
ALLOWED_TYPES = {
    "Tuple", "Union", "bool", "float", "Any", "Dict", "str", "int",
    "set", "dict", "list", "tuple", "Set", "Optional", "List"
}

@dataclass
class OutputDefinition:
    """Definition of a step output."""
    type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OutputDefinition':
        """Create an OutputDefinition from a dictionary."""
        return cls(type=data["type"])

def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    if not name.isidentifier():
        return False
    return True

@dataclass
class Parameter:
    """Parameter definition."""
    name: str
    type: str
    description: str = ""
    
    # Valid Python types that can be used in parameters and outputs
    valid_types = {
        "str", "int", "float", "bool", "list", "dict", "set", "tuple",
        "Any", "Optional", "Union", "List", "Dict", "Set", "Tuple"
    }

@dataclass
class PlanStep:
    """A step in the execution plan."""
    name: str
    tool: str
    inputs: Dict[str, Any]
    outputs: Dict[str, OutputDefinition]
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)
    allowed_create_paths: List[str] = field(default_factory=list)
    default_policy: str = "deny"
    interface_validation: Optional['StepValidationBase'] = None
    script_file: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        """Create a PlanStep from a dictionary."""
        outputs = {
            name: OutputDefinition.from_dict(output_def)
            for name, output_def in data["outputs"].items()
        }
        return cls(
            name=data["name"],
            tool=data["tool"],
            inputs=data["inputs"],
            outputs=outputs,
            allowed_read_paths=data.get("allowed_read_paths", []),
            allowed_write_paths=data.get("allowed_write_paths", []),
            allowed_create_paths=data.get("allowed_create_paths", []),
            default_policy=data.get("default_policy", "deny"),
            script_file=data.get("script_file")
        )

@dataclass
class StepValidationBase:
    """Base class for step validation results."""
    matches_interface: bool = False
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class SystemToolValidation(StepValidationBase):
    """Validation results for system tool steps."""
    matches_interface: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    @classmethod
    def from_step(cls, step: PlanStep, tool: 'SystemTool') -> 'SystemToolValidation':
        """Create a validation instance from a plan step and system tool."""
        validation = cls()
        validation.matches_interface = True
        validation.validation_errors = []
        
        # Validate required inputs
        required_inputs = {param.name for param in tool.interface.parameters}
        missing_inputs = required_inputs - set(step.inputs.keys())
        if missing_inputs:
            validation.matches_interface = False
            validation.validation_errors.append(f"Missing required inputs: {missing_inputs}")
        
        # Validate outputs
        if not step.outputs:
            validation.matches_interface = False
            validation.validation_errors.append("No outputs defined")
        elif len(step.outputs) > 1:
            validation.matches_interface = False
            validation.validation_errors.append("System tools can only have one output")
        else:
            output_type = next(iter(step.outputs.values())).get("type")
            if output_type != tool.interface.return_type:
                validation.matches_interface = False
                validation.validation_errors.append(f"Output type mismatch: expected {tool.interface.return_type}, got {output_type}")
        
        return validation

@dataclass
class GenerateCodeValidation(StepValidationBase):
    """Validation results for generate_code steps."""
    matches_interface: bool = False
    validation_errors: List[str] = field(default_factory=list)

    @classmethod
    def from_step(cls, step: 'PlanStep') -> 'GenerateCodeValidation':
        """Create validation results from a plan step."""
        validation = cls()
        
        # Check required inputs
        required_inputs = {"function_name", "parameters", "return_type", "description"}
        missing_inputs = required_inputs - set(step.inputs.keys())
        if missing_inputs:
            validation.validation_errors.append(
                f"Missing required inputs: {', '.join(missing_inputs)}"
            )
            return validation
            
        # Check function name is valid Python identifier
        if not step.inputs["function_name"].isidentifier():
            validation.validation_errors.append(
                f"Invalid function name: {step.inputs['function_name']}"
            )
            
        # Check parameters are valid
        for param in step.inputs["parameters"]:
            if not param["name"].isidentifier():
                validation.validation_errors.append(
                    f"Invalid parameter name: {param['name']}"
                )
                
        # Check return type is valid
        if step.inputs["return_type"] not in ALLOWED_TYPES:
            validation.validation_errors.append(
                f"Invalid return type: {step.inputs['return_type']}"
            )
            
        # Check outputs include code_file with type str
        if "code_file" not in step.outputs:
            validation.validation_errors.append("Missing required output: code_file")
        elif step.outputs["code_file"].type != "str":
            validation.validation_errors.append(
                f"code_file output must be type str, got {step.outputs['code_file'].type}"
            )
            
        # Set matches_interface if no errors
        validation.matches_interface = len(validation.validation_errors) == 0
        
        return validation

@dataclass
class ExecuteCodeValidation(StepValidationBase):
    """Validation results for execute_code steps."""
    matches_interface: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    @classmethod
    def from_step(cls, step: PlanStep) -> 'ExecuteCodeValidation':
        """Create a validation instance from a plan step."""
        validation = cls()
        validation.matches_interface = True
        validation.validation_errors = []
        
        # Validate required inputs
        if "script_file" not in step.inputs:
            validation.matches_interface = False
            validation.validation_errors.append("Missing required input 'script_file'")
        elif not isinstance(step.inputs["script_file"], str):
            validation.matches_interface = False
            validation.validation_errors.append("Input 'script_file' must be a string")
        elif not step.inputs["script_file"].endswith(".py"):
            validation.matches_interface = False
            validation.validation_errors.append("Input 'script_file' must be a Python file")
        
        return validation

@dataclass
class InterfaceValidation(StepValidationBase):
    """Validation results for interface checks."""
    pass

@dataclass
class ValidationResults:
    """Validation results."""
    syntax_valid: bool
    security_issues: List[str]

@dataclass
class GeneratedCode:
    """Generated code with validation results and outputs."""
    code: str
    validation_results: ValidationResults
    outputs: Dict[str, OutputDefinition]

@dataclass
class FunctionInterface:
    """Function interface specification."""
    outputs: Dict[str, OutputDefinition]

@dataclass
class Plan:
    """Execution plan."""
    steps: List[PlanStep]
    artifacts_dir: str
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary."""
        return {
            "steps": [
                {
                    "name": step.name,
                    "tool": step.tool,
                    "inputs": step.inputs,
                    "outputs": step.outputs,
                    "allowed_read_paths": step.allowed_read_paths,
                    "allowed_write_paths": step.allowed_write_paths,
                    "allowed_create_paths": step.allowed_create_paths,
                    "default_policy": step.default_policy,
                    "interface_validation": {
                        "matches_interface": step.interface_validation.matches_interface if step.interface_validation else False,
                        "validation_errors": step.interface_validation.validation_errors if step.interface_validation else []
                    } if step.interface_validation else None,
                    "script_file": step.script_file
                }
                for step in self.steps
            ],
            "artifacts_dir": self.artifacts_dir
        }

@dataclass
class CodeGenerationRequest:
    """Request to generate code."""
    def __init__(self, function_name: str, parameters: List[Parameter], return_type: str,
                 description: str, prompt: Optional[str] = None, interface: Optional[FunctionInterface] = None,
                 system_prompt: Optional[str] = None, temperature: Optional[float] = None,
                 examples: Optional[List[str]] = None, constraints: Optional[List[str]] = None,
                 example_format: Optional[str] = None):
        self.function_name = function_name
        self.parameters = parameters
        self.return_type = return_type
        self.description = description
        self.prompt = prompt
        self.interface = interface
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.examples = examples or []
        self.constraints = constraints or []
        self.example_format = example_format

@dataclass
class CodeGenerationResponse:
    """Code generation response with validation results and outputs."""
    code: str
    validation_results: ValidationResults
    outputs: Dict[str, OutputDefinition]
    function_name: str
    parameters: List[Parameter]
    return_type: str
    description: str

@dataclass
class ExecutionRequest:
    """Execution request."""
    plan: Plan
    keep_artifacts: bool = False

@dataclass
class ExecutionResponse:
    """Execution response."""
    results: Dict[str, Any]
    artifacts_dir: Optional[str] = None

@dataclass
class TestCase:
    """Test case definition."""
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    description: str = ""

@dataclass
class TestFailure:
    """Test failure details."""
    test_name: str
    expected: Any
    actual: Any
    message: str

@dataclass
class TestResults:
    """Test results."""
    passed: bool
    errors: List[str]
    failures: List[TestFailure] = field(default_factory=list)
    actual_outputs: Optional[Dict[str, Any]] = None

@dataclass
class SystemTool:
    """System tool definition."""
    name: str
    description: str
    parameters: List[Parameter]
    outputs: Dict[str, OutputDefinition]
    permissions: Optional[Dict[str, List[str]]] = None
    filepath: Optional[str] = None

@dataclass
class CodeResponse:
    """Response from code generation."""
    code: str
    function_name: str
    parameters: List[Parameter]
    return_type: str
    description: str
    examples: Optional[List[str]] = None
    constraints: Optional[List[str]] = None

@dataclass
class ToolParameter:
    """Parameter definition for system tools."""
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Optional[Any] = None

@dataclass
class ToolInterface:
    """Interface definition for system tools."""
    function_name: str
    parameters: List[ToolParameter]
    return_type: str
    description: str
    examples: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[str]] = None