from typing import List
from src.chunking.models import CodeChunk


SYSTEM_PROMPT = """You are an expert code assistant that answers questions about codebases.

You will be given:
1. A question about a codebase
2. Relevant code chunks retrieved from that codebase

Your job:
- Answer the question clearly and accurately based on the provided code chunks
- Always cite the exact file path and line numbers for every piece of code you reference
- Format citations as: `filename.py` (lines X-Y)
- If the answer is not in the provided chunks, say "I couldn't find relevant code for this question"
- Keep answers concise but complete
- Use markdown formatting for code blocks

Never make up code that isn't in the provided chunks."""


def build_context(chunks: List[CodeChunk]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        name = chunk.function_name or chunk.class_name or "block"
        header = (
            f"--- Chunk {i} ---\n"
            f"File: {chunk.file_path}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Type: {chunk.chunk_type}\n"
            f"Name: {name}\n"
        )
        truncated = chunk.content[:800]
        context_parts.append(f"{header}\n```{chunk.language}\n{truncated}\n```")

    return "\n\n".join(context_parts)


def build_prompt(query: str, chunks: List[CodeChunk]) -> str:
    """Build the full user prompt with context and question."""
    context = build_context(chunks)

    prompt = f"""Here are the relevant code chunks from the codebase:

{context}

---

Question: {query}

Please answer based on the code chunks above. Cite file paths and line numbers."""

    return prompt