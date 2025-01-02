def main(inputs, output_dir):
    """Simple test system tool that just returns a dummy result"""
    csv_path = inputs.get("csv_path")
    if not csv_path:
        raise ValueError("Missing required input: csv_path")

    # In a real tool, we would process the CSV here
    # For now, just return a dummy result
    return {"cleaned_csv_path": f"{output_dir}/cleaned.csv"}
