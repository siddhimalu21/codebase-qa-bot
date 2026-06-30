from typing import List, Dict
from src.chunking.models import CodeChunk
from src.retrieval.vector_retriever import vector_search
from src.retrieval.bm25_retriever import BM25Retriever


def reciprocal_rank_fusion(
    vector_results: List[CodeChunk],
    bm25_results: List[CodeChunk],
    k: int = 60,
) -> List[CodeChunk]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = 1/(k + rank) for each result in each list.
    Higher score = more relevant.

    k=60 is the standard value from the original RRF paper.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, CodeChunk] = {}

    # Score vector results
    for rank, chunk in enumerate(vector_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    # Sort by combined score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


def hybrid_search(
    query: str,
    repo_name: str,
    all_chunks: List[CodeChunk],
    top_k: int = 10,
) -> List[CodeChunk]:
    """
    Hybrid search combining vector similarity and BM25 keyword search.

    Args:
        query: Natural language question
        repo_name: Name of the indexed repo
        all_chunks: All chunks (needed for BM25 index)
        top_k: Number of final results to return

    Returns:
        Fused and ranked list of CodeChunk objects
    """
    # Vector search
    vector_results = vector_search(query, repo_name, top_k=top_k)

    # BM25 search
    bm25 = BM25Retriever(all_chunks)
    bm25_results = bm25.search(query, top_k=top_k)

    # Fuse results
    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    return fused[:top_k]