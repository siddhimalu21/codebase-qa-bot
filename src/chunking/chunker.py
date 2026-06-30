from typing import List
from src.ingestion.models import CodeFile
from src.chunking.models import CodeChunk
from src.chunking.python_chunker import chunk_python_file
from src.chunking.generic_chunker import chunk_generic_file


def chunk_code_file(code_file: CodeFile) -> List[CodeChunk]:
    """
    Route a CodeFile to the right chunker based on language.
    Python gets AST-aware chunking.
    Everything else gets line-based chunking.
    """
    if code_file.language == "python":
        return chunk_python_file(code_file)
    else:
        return chunk_generic_file(code_file)


def chunk_all_files(code_files: List[CodeFile]) -> List[CodeChunk]:
    """
    Chunk all files in a repo and return a flat list of CodeChunks.
    """
    all_chunks = []
    total_files = len(code_files)

    for i, code_file in enumerate(code_files, 1):
        chunks = chunk_code_file(code_file)
        all_chunks.extend(chunks)
        print(f"  [{i}/{total_files}] {code_file.file_path} -> {len(chunks)} chunks")

    return all_chunks