from typing import List
from src.ingestion.models import CodeFile
from src.chunking.models import CodeChunk


def _make_chunk_id(repo_name: str, file_path: str, index: int) -> str:
    clean_path = file_path.replace("\\", "_").replace("/", "_")
    return f"{repo_name}__{clean_path}__block__{index}"


def chunk_generic_file(code_file: CodeFile, chunk_size: int = 50) -> List[CodeChunk]:
    """
    Line-based chunker for non-Python files (JS, TS, Java, Go etc).
    Splits every chunk_size lines with a 10-line overlap.
    """
    lines = code_file.content.splitlines()
    chunks = []
    index = 0
    overlap = 10
    i = 0

    while i < len(lines):
        chunk_lines = lines[i: i + chunk_size]
        content = "\n".join(chunk_lines)

        if content.strip():
            chunk = CodeChunk(
                chunk_id=_make_chunk_id(
                    code_file.repo_name,
                    code_file.file_path,
                    index
                ),
                content=content,
                chunk_type="block",
                file_path=code_file.file_path,
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                language=code_file.language,
                repo_name=code_file.repo_name,
                repo_url=code_file.repo_url,
            )
            chunks.append(chunk)
            index += 1

        i += chunk_size - overlap

    return chunks