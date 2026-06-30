import ast
from typing import List
from src.ingestion.models import CodeFile
from src.chunking.models import CodeChunk


def _make_chunk_id(repo_name: str, file_path: str, name: str, index: int) -> str:
    """Create a unique chunk ID."""
    clean_path = file_path.replace("\\", "_").replace("/", "_")
    clean_name = name.replace(" ", "_")
    return f"{repo_name}__{clean_path}__{clean_name}__{index}"


def _extract_node_source(source_lines: list, node: ast.AST) -> str:
    """Extract source code for an AST node using line numbers."""
    start = node.lineno - 1
    end = node.end_lineno
    return "\n".join(source_lines[start:end])


def chunk_python_file(code_file: CodeFile) -> List[CodeChunk]:
    """
    Parse a Python file using AST and split into chunks
    at function and class boundaries.
    """
    chunks = []
    source = code_file.content
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _fallback_chunk(code_file)

    index = 0

    for node in ast.walk(tree):
        # Handle top-level and nested functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_content = _extract_node_source(source_lines, node)

            if not chunk_content.strip():
                continue

            # Detect if this function is inside a class
            class_name = ""
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.walk(parent):
                        if child is node:
                            class_name = parent.name
                            break

            chunk = CodeChunk(
                chunk_id=_make_chunk_id(
                    code_file.repo_name,
                    code_file.file_path,
                    node.name,
                    index
                ),
                content=chunk_content,
                chunk_type="function",
                file_path=code_file.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                language=code_file.language,
                repo_name=code_file.repo_name,
                repo_url=code_file.repo_url,
                function_name=node.name,
                class_name=class_name,
            )
            chunks.append(chunk)
            index += 1

        # Handle class definitions
        elif isinstance(node, ast.ClassDef):
            chunk_content = _extract_node_source(source_lines, node)

            if not chunk_content.strip():
                continue

            chunk = CodeChunk(
                chunk_id=_make_chunk_id(
                    code_file.repo_name,
                    code_file.file_path,
                    node.name,
                    index
                ),
                content=chunk_content,
                chunk_type="class",
                file_path=code_file.file_path,
                start_line=node.lineno,
                end_line=node.end_lineno,
                language=code_file.language,
                repo_name=code_file.repo_name,
                repo_url=code_file.repo_url,
                class_name=node.name,
            )
            chunks.append(chunk)
            index += 1

    # If AST found nothing, fall back to line-based chunking
    if not chunks:
        return _fallback_chunk(code_file)

    return chunks


def _fallback_chunk(code_file: CodeFile, chunk_size: int = 50) -> List[CodeChunk]:
    """Fall back to line-based chunking if AST parsing fails."""
    lines = code_file.content.splitlines()
    chunks = []
    index = 0

    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i: i + chunk_size]
        content = "\n".join(chunk_lines)

        if not content.strip():
            continue

        chunk = CodeChunk(
            chunk_id=_make_chunk_id(
                code_file.repo_name,
                code_file.file_path,
                "block",
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

    return chunks