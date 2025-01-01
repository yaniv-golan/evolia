import pytest
from evolia.models import Parameter, FunctionInterface
from evolia.core.interface_verification import verify_tool_interface
from typing import Dict, Any

def test_function_parameter_validation():
    # Test valid parameter
    param = Parameter(
        name="input_path",
        type="str",
        description="Path to the input file"
    )
    assert param.name == "input_path"
    assert param.type == "str"
    assert param.description == "Path to the input file"
    
    # Test missing required fields
    with pytest.raises(ValueError):
        Parameter(name="input_path")
    
    with pytest.raises(ValueError):
        Parameter(type="str")

def test_function_interface_validation():
    # Test valid interface
    interface = FunctionInterface(
        function_name="process_data",
        parameters=[
            Parameter(
                name="input_path",
                type="str",
                description="Path to the input file"
            )
        ],
        return_type="str",
        description="Process the input file",
        examples=["process_data('input.csv')"],
        constraints=["Input must be CSV"]
    )
    
    assert interface.function_name == "process_data"
    assert len(interface.parameters) == 1
    assert interface.return_type == "str"
    assert interface.description == "Process the input file"
    assert interface.examples == ["process_data('input.csv')"]
    assert interface.constraints == ["Input must be CSV"]
    
    # Test missing required fields
    with pytest.raises(ValueError):
        FunctionInterface(
            function_name="process_data",
            return_type="str"
        )

def test_verify_interface():
    """Test interface verification logic"""
    interface = FunctionInterface(
        function_name="process_data",
        parameters=[
            Parameter(name="input_path", type="str", description="Path to input file")
        ],
        return_type="str",
        description="Process data from a file"
    )
    
    generated_code = {
        "function_name": "wrong_name",
        "parameters": [
            {"name": "input_path", "type": "str"}
        ],
        "return_type": "str"
    }
    
    # Test function name mismatch
    errors = verify_tool_interface(generated_code, interface)
    assert len(errors) == 1
    assert "Function name mismatch" in errors[0]
    
    # Test mismatched parameter type
    generated_code["function_name"] = "process_data"
    generated_code["parameters"][0]["type"] = "int"
    errors = verify_tool_interface(generated_code, interface)
    assert len(errors) == 1
    assert "Parameter mismatch" in errors[0]
    
    # Test mismatched return type
    generated_code["parameters"][0]["type"] = "str"
    generated_code["return_type"] = "int"
    errors = verify_tool_interface(generated_code, interface)
    assert len(errors) == 1
    assert "Return type mismatch" in errors[0]
    
    # Test missing parameter
    generated_code["return_type"] = "str"
    generated_code["parameters"] = []
    errors = verify_tool_interface(generated_code, interface)
    assert len(errors) == 1
    assert "Parameter mismatch" in errors[0] 