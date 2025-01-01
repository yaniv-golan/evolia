import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class CandidateManager:
    def __init__(self, base_dir: str = "run_artifacts"):
        # Convert base_dir to absolute path
        self.base_dir = Path(base_dir).resolve()
        self.tmp_dir = self.base_dir / "tmp"
        self.candidates_dir = self.base_dir / "candidates"
        self.candidates_json = self.candidates_dir / "candidates.json"
        
        # Ensure directories exist
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize candidates.json if it doesn't exist
        if not self.candidates_json.exists():
            with open(self.candidates_json, 'w') as f:
                json.dump([], f)
        else:
            # Ensure the file contains a valid array
            try:
                with open(self.candidates_json) as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        with open(self.candidates_json, 'w') as f:
                            json.dump([], f)
            except (json.JSONDecodeError, FileNotFoundError):
                with open(self.candidates_json, 'w') as f:
                    json.dump([], f)

    def load_candidates(self) -> List[Dict]:
        """Load the candidates metadata from candidates.json."""
        try:
            with open(self.candidates_json) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def save_candidates(self, candidates: List[Dict]) -> None:
        """Save the candidates metadata to candidates.json."""
        with open(self.candidates_json, 'w') as f:
            json.dump(candidates, f, indent=2)

    def move_to_candidates(self, ephemeral_file_path: str, auto_promote: bool = False) -> str:
        """
        Move an ephemeral file to the candidates directory and track it.
        
        Args:
            ephemeral_file_path: Path to the ephemeral file
            auto_promote: Whether this candidate should be auto-promoted when thresholds are met
            
        Returns:
            The path to the new candidate file
        """
        src_path = Path(ephemeral_file_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Ephemeral file not found: {ephemeral_file_path}")
            
        # Generate unique candidate name
        candidate_name = f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dst_path = self.candidates_dir.resolve() / f"{candidate_name}.py"
        
        # Move the file
        with open(src_path, 'r', encoding='utf-8') as src, open(dst_path, 'w', encoding='utf-8') as dst:
            content = src.read().strip()
            dst.write(content + '\n')
        os.remove(str(src_path))
        
        # Add to candidates.json
        candidates = self.load_candidates()
        candidates.append({
            "name": candidate_name,
            "filepath": str(dst_path),
            "usage_count": 0,
            "success_count": 0,
            "auto_promote": auto_promote,
            "date_created": datetime.now().isoformat()
        })
        self.save_candidates(candidates)
        
        return str(dst_path)

    def update_candidate_usage(self, candidate_path: str, success: bool = True) -> None:
        """Update usage statistics for a candidate tool."""
        candidates = self.load_candidates()
        abs_path = str(Path(candidate_path).resolve())
        
        for candidate in candidates:
            if candidate["filepath"] == abs_path:
                candidate["usage_count"] += 1
                if success:
                    candidate["success_count"] += 1
                break
                
        self.save_candidates(candidates)

    def get_candidate_stats(self, candidate_name: str) -> Optional[Dict]:
        """Get usage statistics for a candidate by name."""
        candidates = self.load_candidates()
        for candidate in candidates:
            if candidate["name"] == candidate_name:
                return candidate
        return None

    def check_promotion_eligibility(self, 
                                  usage_threshold: int = 3, 
                                  success_ratio_threshold: float = 0.8) -> List[str]:
        """
        Check which candidates are eligible for promotion based on thresholds.
        
        Returns:
            List of candidate names eligible for promotion
        """
        eligible = []
        candidates = self.load_candidates()
        
        for candidate in candidates:
            if not candidate.get("auto_promote", False):
                continue
                
            usage = candidate["usage_count"]
            success = candidate["success_count"]
            
            if usage >= usage_threshold and (success / usage) >= success_ratio_threshold:
                eligible.append(candidate["name"])
                
        return eligible 