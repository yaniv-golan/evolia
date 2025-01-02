from typing import List

def calculate_average(numbers: List[int]) -> float:
    """Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers
        
    Returns:
        The average of the numbers
    """
    return sum(numbers) / len(numbers)