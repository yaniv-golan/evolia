from typing import List

def create_number_list(start: int, end: int) -> List[int]:
    """Create a list of numbers from start to end.
    
    Args:
        start: Start number
        end: End number
        
    Returns:
        The list of numbers
    """
    return list(range(start, end + 1))