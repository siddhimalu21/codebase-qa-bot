from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeFile:
    """Represents a single source code file loaded from a repository."""

    # Core content
    content: str                    # raw file content as string
    file_path: str                  # relative path inside repo e.g. src/auth/login.py
    language: str                   # detected language e.g. python, javascript

    # Repository metadata
    repo_name: str                  # e.g. flask
    repo_url: str                   # original GitHub URL

    # File metadata
    file_size: int = 0              # bytes
    line_count: int = 0             # total lines in file
    extension: str = ""             # e.g. .py

    def __post_init__(self):
        """Auto-calculate derived fields after init."""
        self.line_count = len(self.content.splitlines())
        self.file_size = len(self.content.encode("utf-8"))

    def __repr__(self):
        return (
            f"CodeFile(path={self.file_path}, "
            f"lang={self.language}, "
            f"lines={self.line_count})"
        )