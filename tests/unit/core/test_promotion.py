"""Tests for tool promotion functionality."""
import pytest
from pathlib import Path
from evolia.core.promotion import ToolPromoter

def test_prompt_for_promotion(tmp_path):
    """Test tool promotion functionality."""
    # Create test directories
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    
    # Create a candidate tool file
    candidate_file = tools_dir / "candidate_add.py"
    candidate_file.write_text("def add(x: int, y: int) -> int:\n    return x + y")
    
    # Create metadata for the candidate
    candidate_metadata = {
        "name": "candidate_add",
        "usage_count": 10,
        "success_count": 8
    }
    
    # Create promoter with test paths
    promoter = ToolPromoter(
        system_tools_dir=str(tools_dir / "system"),
        system_tools_json=str(tools_dir / "system_tools.json")
    )
    
    # Test promotion
    promoted_path = promoter.promote_candidate_to_system(
        str(candidate_file),
        candidate_metadata,
        "A tool that adds two numbers"
    )
    
    # Verify promotion results
    assert promoted_path is not None
    assert Path(promoted_path).exists()
    
    # Verify tool metadata was recorded
    tool = promoter.get_system_tool("tool_add")
    assert tool is not None
    assert tool["description"] == "A tool that adds two numbers"
    assert tool["promotion_stats"]["usage_count"] == 10
    assert tool["promotion_stats"]["success_count"] == 8
    assert tool["promotion_stats"]["success_ratio"] == 0.8

    # Test promoting non-existent file
    with pytest.raises(FileNotFoundError):
        promoter.promote_candidate_to_system(
            "nonexistent.py",
            {"name": "nonexistent"},
            "This should fail"
        ) 