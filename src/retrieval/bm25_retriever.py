from typing import List
from rank_bm25 import BM25Okapi
from src.chunking.models import CodeChunk


class BM25Retriever:
    """
    Keyword-based retriever using BM25 algorithm.
    Great for exact matches like function names,
    variable names, and specific keywords.
    """

    def __init__(self, chunks: List[CodeChunk]):
        self.chunks = chunks
        # Tokenize each chunk by splitting on whitespace and special chars
        tokenized = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer — split on whitespace and common code separators."""
        import re
        tokens = re.split(r"[\s\(\)\[\]\{\}.,;:=<>!&|+\-*/\\\"'`]+", text.lower())
        return [t for t in tokens if t and len(t) > 1]

    def search(self, query: str, top_k: int = 10) -> List[CodeChunk]:
        """
        Search chunks using BM25 keyword matching.

        Returns:
            List of CodeChunk objects ranked by BM25 score
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top_k highest scores
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self.chunks[idx])

        return results