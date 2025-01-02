"""Interface verification methods for executor2."""

import ast
from typing import Dict, Any, List, Optional
from ..models.models import FunctionInterface

def verify_interface(interface: FunctionInterface, generated_code: Dict[str, Any]) -> List[str]:
    """Verify that generated code matches the interface specification.
    
    Args:
        interface: Interface specification to verify against
        generated_code: Generated code response with validation results
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check code exists
    if 'code' not in generated_code:
        errors.append("Generated code missing 'code' field")
        return errors
        
    # Check function info exists
    if 'function_info' not in generated_code:
        errors.append("Generated code missing 'function_info' field")
        return errors
        
    # Get function info
    function_info = generated_code['function_info']
    
    # Check function name matches
    if function_info['name'] != interface.function_name:
        errors.append(f"Function name mismatch: expected {interface.function_name}, got {function_info['name']}")
    
    # Check parameters match
    interface_params = {p.name: p.type for p in interface.parameters}
    generated_params = {p['name']: p['type'] for p in function_info['parameters']}
    
    if interface_params != generated_params:
        errors.append(f"Parameter mismatch: expected {interface_params}, got {generated_params}")
    
    # Check return type matches
    if function_info['return_type'] != interface.return_type:
        errors.append(f"Return type mismatch: expected {interface.return_type}, got {function_info['return_type']}")
    
    return errors

# Alias for backward compatibility
verify_tool_interface = verify_interface

def match_example(generated_example: Dict[str, Any], interface_example: Dict[str, Any]) -> bool:
    """Check if a generated example matches an interface example.
    
    Args:
        generated_example: Example from generated code
        interface_example: Example from interface
        
    Returns:
        bool: Whether the examples match
    """
    try:
        # Check inputs match
        if not match_dict_structure(generated_example.get('inputs', {}), 
                                        interface_example.get('inputs', {})):
            return False
            
        # Check expected outputs match
        if not match_dict_structure(generated_example.get('expected', {}),
                                        interface_example.get('expected', {})):
            return False
            
        return True
    except Exception:
        return False

def match_dict_structure(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> bool:
    """Compare dictionary structures recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        bool: Whether the structures match
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return False
        
    if set(dict1.keys()) != set(dict2.keys()):
        return False
        
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not match_dict_structure(dict1[key], dict2[key]):
                return False
        elif type(dict1[key]) != type(dict2[key]):
            return False
            
    return True

def verify_constraint(code: str, constraint: str) -> bool:
    """Verify a code constraint is met.
    
    Args:
        code: The generated code
        constraint: The constraint to verify
        
    Returns:
        bool: Whether the constraint is met
    """
    try:
        tree = ast.parse(code)
        
        if constraint == 'no_globals':
            # Check for global variable assignments
            return not any(isinstance(node, ast.Global) for node in ast.walk(tree))
            
        elif constraint == 'no_nested_functions':
            # Check for nested function definitions
            function_depth = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_depth += 1
                    if function_depth > 1:
                        return False
            return True
            
        elif constraint == 'use_type_hints':
            # Check all function arguments have type hints
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not all(arg.annotation for arg in node.args.args):
                        return False
            return True
            
        elif constraint == 'handle_errors':
            # Check for try/except blocks
            return any(isinstance(node, ast.Try) for node in ast.walk(tree))
            
        elif constraint == 'return_dict':
            # Check all return values are dictionaries
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    if not isinstance(node.value, ast.Dict):
                        return False
            return True
            
        # Add more constraints as needed
            
        return True  # Unknown constraint passes by default
        
    except Exception:
        return False 