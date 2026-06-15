"""
State tracking for S3 uploads — avoid re-uploading unchanged files.

This module tracks file modification times and sizes to implement intelligent
change detection. Only modified/new files are uploaded in subsequent runs.

Storage: ~/.scenario_reasoner_lm/upload_state.json (persistent across runs)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class UploadState:
    """Track uploaded files by mtime and size to enable incremental uploads."""
    
    def __init__(self, state_dir: Optional[Path] = None) -> None:
        """
        Initialize upload state.
        
        Args:
            state_dir: Directory to store state file. Defaults to ~/.scenario_reasoner_lm/
        """
        if state_dir is None:
            state_dir = Path.home() / ".scenario_reasoner_lm"
        
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / "upload_state.json"
        self.state: Dict[str, Dict[str, Any]] = self._load()
    
    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Load state from disk."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: Failed to load state: {e}. Starting fresh.")
                return {}
        return {}
    
    def save(self) -> None:
        """Save state to disk."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
    
    def _file_signature(self, path: Path) -> tuple[float, int]:
        """Get file modification time and size as unique signature."""
        stat = path.stat()
        return (stat.st_mtime, stat.st_size)
    
    def needs_upload(self, path: Path) -> bool:
        """
        Check if file has changed since last upload.
        
        Args:
            path: Absolute path to file
            
        Returns:
            True if file is new or has changed, False if unchanged
        """
        path = path.resolve()
        key = str(path)
        
        mtime, size = self._file_signature(path)
        prev_state = self.state.get(key)
        
        if prev_state is None:
            # Never uploaded before
            return True
        
        # File unchanged if both mtime and size match
        return (
            prev_state.get("mtime") != mtime
            or prev_state.get("size") != size
        )
    
    def mark_uploaded(
        self,
        path: Path,
        s3_key: str,
        compressed: bool = False,
    ) -> None:
        """
        Record that a file was successfully uploaded.
        
        Args:
            path: Absolute path to local file
            s3_key: S3 key (path in bucket) where file was uploaded
            compressed: Whether file was gzipped before upload
        """
        path = path.resolve()
        key = str(path)
        
        mtime, size = self._file_signature(path)
        
        self.state[key] = {
            "mtime": mtime,
            "size": size,
            "s3_key": s3_key,
            "compressed": compressed,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def get_uploaded_key(self, path: Path) -> Optional[str]:
        """Get the S3 key where this file was uploaded."""
        path = path.resolve()
        key = str(path)
        prev_state = self.state.get(key)
        return prev_state.get("s3_key") if prev_state else None
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Get summary statistics of tracked uploads."""
        return {
            "total_files": len(self.state),
            "last_updated": (
                max(
                    s["uploaded_at"] for s in self.state.values()
                    if "uploaded_at" in s
                )
                if self.state
                else None
            ),
        }
    
    def clear(self) -> None:
        """Clear all tracked state (force re-upload of all files)."""
        self.state.clear()
        self.save()


def get_state(state_dir: Optional[Path] = None) -> UploadState:
    """Get or create global upload state instance."""
    return UploadState(state_dir)
