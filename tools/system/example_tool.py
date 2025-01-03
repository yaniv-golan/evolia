"""Example tool that demonstrates the enhanced metadata structure"""

from pathlib import Path

import pandas as pd


def process_data(input_path: str, output_dir: str) -> str:
    """Process a CSV file and return the path to the processed output.

    Args:
        input_path: Path to the input CSV file
        output_dir: Directory where the output file should be saved

    Returns:
        str: Path to the processed output file
    """
    # Read the input CSV
    df = pd.read_csv(input_path)

    # Do some processing (example: remove duplicates)
    df = df.drop_duplicates()

    # Save to output file
    output_path = str(Path(output_dir) / "processed.csv")
    df.to_csv(output_path, index=False)

    return output_path


def main(inputs: dict, output_dir: str) -> dict:
    """Main entry point for the tool.

    Args:
        inputs: Dictionary containing input parameters
        output_dir: Directory for output files

    Returns:
        dict: Dictionary containing output paths
    """
    input_path = inputs["input_path"]
    output_path = process_data(input_path, output_dir)
    return {"output_path": output_path}
