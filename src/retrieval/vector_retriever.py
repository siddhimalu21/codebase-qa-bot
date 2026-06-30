from typing import List
from src.chunking.models import CodeChunk
from src.embeddings.embedder import get_embedder
from src.embeddings.vector_store import query_collection


def vector_search(
    query: str,
    repo_name: str,
    top_k: int = 10,
) -> List[CodeChunk]:
    """
    Search ChromaDB using semantic vector similarity.

    Args:
        query: Natural language question
        repo_name: Name of the indexed repo
        top_k: Number of results to return

    Returns:
        List of CodeChunk objects ranked by similarity
    """
    embedder = get_embedder()
    query_embedding = embedder.embed_text(query)

    results = query_collection(
        query_embedding=query_embedding,
        repo_name=repo_name,
        top_k=top_k,
    )

    chunks = []
    if not results or not results.get("documents"):
        return chunks

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunk = CodeChunk(
            chunk_id=meta["chunk_id"],
            content=doc,
            chunk_type=meta["chunk_type"],
            file_path=meta["file_path"],
            start_line=meta["start_line"],
            end_line=meta["end_line"],
            language=meta["language"],
            repo_name=meta["repo_name"],
            repo_url=meta["repo_url"],
            function_name=meta.get("function_name", ""),
            class_name=meta.get("class_name", ""),
        )
        chunks.append(chunk)

    return chunks