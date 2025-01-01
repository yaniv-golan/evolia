import json
from pathlib import Path
import pytest
from evolia.models import SystemTool

def test_system_tools_json():
    """Test that system tools JSON is valid and matches expected format"""
    tools_path = Path(__file__).parent.parent / "data" / "system_tools.json"
    
    with open(tools_path) as f:
        tools_data = json.load(f)
    
    for tool_data in tools_data:
        try:
            # Verify required fields
            assert "name" in tool_data, "Missing name field"
            assert "filepath" in tool_data, "Missing filepath field"
            assert "version" in tool_data, "Missing version field"
            assert "interface" in tool_data, "Missing interface field"
            
            # Verify interface fields
            interface = tool_data["interface"]
            assert "function_name" in interface, "Missing function_name in interface"
            assert "parameters" in interface, "Missing parameters in interface"
            assert "return_type" in interface, "Missing return_type in interface"
            
            # Verify parameter format
            for param in interface["parameters"]:
                assert "name" in param, "Missing parameter name"
                assert "type" in param, "Missing parameter type"
                assert "description" in param, "Missing parameter description"
                
                # Verify parameter values
                if param["name"] == "input_path":
                    assert param["type"] == "str"
                    assert param["description"] == "Path to the input file. Must be a CSV file."
                
            # Verify return type
            assert interface["return_type"] in ["str", "int", "float", "bool", "list", "dict", "None"]
            
        except AssertionError as e:
            pytest.fail(f"Failed to validate tool {tool_data.get('name', 'unknown')}: {str(e)}") 