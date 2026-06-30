from dataclasses import dataclass


@dataclass
class CodeChunk:
    """Represents a single chunk of code ready for embedding."""

    # Core content
    chunk_id: str           # unique id e.g. flask__app.py__authenticate__0
    content: str            # the actual code text
    chunk_type: str         # 'function', 'class', 'module', 'block'

    # Source location
    file_path: str          # relative path in repo
    start_line: int         # line where this chunk starts
    end_line: int           # line where this chunk ends

    # Semantic metadata
    language: str           # python, javascript etc
    repo_name: str          # e.g. flask
    repo_url: str           # original GitHub URL
    function_name: str = "" # name if chunk is a function
    class_name: str = ""    # name if chunk is inside a class

    # Stats
    line_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        self.line_count = len(self.content.splitlines())
        self.char_count = len(self.content)

    def __repr__(self):
        name = self.function_name or self.class_name or "module"
        return (
            f"CodeChunk(id={self.chunk_id}, "
            f"type={self.chunk_type}, "
            f"name={name}, "
            f"lines={self.start_line}-{self.end_line})"
        )

    def to_metadata(self) -> dict:
        """Return metadata dict for storing in vector DB."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "line_count": self.line_count,
            "char_count": self.char_count,
        }