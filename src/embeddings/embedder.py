from sentence_transformers import SentenceTransformer
from typing import List
from src.config import config


class CodeEmbedder:
    """
    Wraps sentence-transformers to embed code chunks.
    Uses all-MiniLM-L6-v2 — free, fast, runs locally.
    Downloads the model once (~90MB) on first run.
    """

    def __init__(self):
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        print("(First run downloads ~90MB — subsequent runs are instant)")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string and return as list of floats."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embed a list of strings in batches.
        Batching prevents memory issues on large repos.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


# Singleton — only load the model once
_embedder_instance = None


def get_embedder() -> CodeEmbedder:
    """Return the shared embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = CodeEmbedder()
    return _embedder_instance