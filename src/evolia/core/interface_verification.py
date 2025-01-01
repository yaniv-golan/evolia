"""Interface verification methods for executor2."""

import ast
from typing import Dict, Any, List, Optional
from ..models.models import FunctionInterface

def verify_interface(generated_code: Dict[str, Any], interface: FunctionInterface) -> List[str]:
    """Verify that generated code matches the interface specification.
    
    Args:
        generated_code: Generated code response with validation results
        interface: Interface specification to verify against
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check code exists
    if 'code' not in generated_code:
        errors.append("Generated code missing 'code' field")
        return errors
        
    # Check validation results exist
    if 'validation_results' not in generated_code:
        errors.append("Generated code missing 'validation_results' field")
        return errors
        
    # Check outputs exist
    if 'outputs' not in generated_code:
        errors.append("Generated code missing 'outputs' field")
        return errors
        
    # Check validation results
    validation_results = generated_code['validation_results']
    if not validation_results.get('syntax_valid', False):
        errors.append("Generated code has syntax errors")
        
    if validation_results.get('security_issues', []):
        errors.extend(validation_results['security_issues'])
        
    # Check outputs match interface
    outputs = generated_code['outputs']
    for output_name, output_info in interface.outputs.items():
        if output_name not in outputs:
            errors.append(f"Missing required output '{output_name}'")
            continue
            
        output_def = outputs[output_name]
        if 'type' not in output_def:
            errors.append(f"Output '{output_name}' missing type definition")
            continue
            
        if output_def['type'] != output_info.type:
            errors.append(
                f"Output '{output_name}' type mismatch: "
                f"expected {output_info.type}, got {output_def['type']}"
            )
            
        if 'reference' not in output_def:
            errors.append(f"Output '{output_name}' missing reference")
            continue
            
        # Validate reference format
        ref = output_def['reference']
        if not ref.startswith('$') or '.' not in ref:
            errors.append(
                f"Output '{output_name}' has invalid reference format: {ref}. "
                "Must be in format $stepname.outputname"
            )
    
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