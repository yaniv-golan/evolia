import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class PromotionError(Exception):
    """Exception raised when tool promotion fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class ToolPromoter:
    def __init__(
        self,
        system_tools_dir: str = "tools/system",
        system_tools_json: str = "data/system_tools.json",
    ):
        self.system_tools_dir = Path(system_tools_dir)
        self.system_tools_json = Path(system_tools_json)

        # Ensure system tools directory exists
        self.system_tools_dir.mkdir(parents=True, exist_ok=True)

        # Initialize system_tools.json if it doesn't exist
        if not self.system_tools_json.exists():
            with open(self.system_tools_json, "w") as f:
                json.dump([], f)

    def load_system_tools(self) -> list:
        """Load the system tools metadata."""
        try:
            with open(self.system_tools_json) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def save_system_tools(self, tools: list) -> None:
        """Save the system tools metadata."""
        with open(self.system_tools_json, "w") as f:
            json.dump(tools, f, indent=2)

    def promote_candidate_to_system(
        self, candidate_path: str, candidate_metadata: Dict, description: str = ""
    ) -> Optional[str]:
        """
        Promote a candidate tool to a system tool.

        Args:
            candidate_path: Path to the candidate file
            candidate_metadata: Metadata about the candidate from candidates.json
            description: Optional description of the tool's functionality

        Returns:
            Path to the new system tool file, or None if promotion failed
        """
        src_path = Path(candidate_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

        # Generate system tool name from candidate name
        system_name = candidate_metadata["name"].replace("candidate_", "tool_")
        dst_path = self.system_tools_dir / f"{system_name}.py"

        # Copy the file to system tools
        shutil.copy2(src_path, dst_path)

        # Add to system_tools.json
        tools = self.load_system_tools()
        tools.append(
            {
                "name": system_name,
                "filepath": str(dst_path.relative_to(self.system_tools_dir.parent)),
                "description": description
                or f"Promoted from candidate {candidate_metadata['name']}",
                "date_promoted": datetime.now().isoformat(),
                "promotion_stats": {
                    "usage_count": candidate_metadata["usage_count"],
                    "success_count": candidate_metadata["success_count"],
                    "success_ratio": candidate_metadata["success_count"]
                    / candidate_metadata["usage_count"],
                },
            }
        )
        self.save_system_tools(tools)

        return str(dst_path)

    def get_system_tool(self, tool_name: str) -> Optional[Dict]:
        """Get metadata for a system tool by name."""
        tools = self.load_system_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return tool
        return None
