

def clean_text(text: str) -> str:
    """Convert text to uppercase and remove all whitespace.
    
    Args:
        text: Input text to process
        
    Returns:
        The cleaned text
    """
    return text.upper().replace(" ", "")