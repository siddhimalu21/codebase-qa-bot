from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.chunking.models import CodeChunk


# This model is ~90MB, downloads once on first run
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CodeReranker:
    """
    Cross-encoder re-ranker that scores query-chunk pairs directly.

    Unlike bi-encoders (which embed query and chunk separately),
    a cross-encoder sees both together — much more accurate scoring.

    Use after retrieval to re-rank top-k chunks before passing to LLM.
    """

    def __init__(self):
        print(f"Loading re-ranker model: {RERANKER_MODEL}")
        print("(First run downloads ~90MB — subsequent runs are instant)")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("Re-ranker loaded.")

    def rerank(
        self,
        query: str,
        chunks: List[CodeChunk],
        top_k: int = 3,
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Re-rank chunks by relevance to the query.

        Args:
            query: The user question
            chunks: Candidate chunks from hybrid search
            top_k: How many to return after re-ranking

        Returns:
            List of (chunk, score) tuples sorted by score descending
        """
        if not chunks:
            return []

        # Build query-document pairs for the cross-encoder
        pairs = [[query, chunk.content] for chunk in chunks]

        # Score all pairs at once
        scores = self.model.predict(pairs)

        # Zip chunks with scores and sort
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def rerank_chunks_only(
        self,
        query: str,
        chunks: List[CodeChunk],
        top_k: int = 3,
    ) -> List[CodeChunk]:
        """
        Re-rank and return just the chunks without scores.
        Convenience method for the QA engine.
        """
        scored = self.rerank(query, chunks, top_k)
        return [chunk for chunk, score in scored]


# Singleton
_reranker_instance = None


def get_reranker() -> CodeReranker:
    """Return the shared reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CodeReranker()
    return _reranker_instance