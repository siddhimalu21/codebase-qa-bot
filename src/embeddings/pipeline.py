from typing import List
from src.chunking.models import CodeChunk
from src.embeddings.embedder import get_embedder
from src.embeddings.vector_store import store_chunks, get_collection_stats


def build_index(chunks: List[CodeChunk], repo_name: str) -> None:
    """
    Full pipeline: embed all chunks and store in ChromaDB.

    Steps:
        1. Extract text content from each chunk
        2. Embed all texts in batches
        3. Store embeddings + metadata in ChromaDB
    """
    print(f"\nBuilding index for '{repo_name}'...")
    print(f"Total chunks to embed: {len(chunks)}")

    # Step 1 — get embedder (loads model on first call)
    embedder = get_embedder()

    # Step 2 — prepare texts for embedding
    # We embed: function/class name + file path + content
    # This gives the model more context than content alone
    texts = []
    for chunk in chunks:
        name = chunk.function_name or chunk.class_name or "block"
        text = f"{name}\n{chunk.file_path}\n{chunk.content}"
        texts.append(text)

    print(f"\nEmbedding {len(texts)} chunks...")
    embeddings = embedder.embed_texts(texts)
    print(f"Embedding complete. Generated {len(embeddings)} vectors.")

    # Step 3 — store in ChromaDB
    store_chunks(chunks, embeddings, repo_name)

    # Step 4 — print stats
    stats = get_collection_stats(repo_name)
    print(f"\nIndex stats:")
    print(f"  Collection : {stats['collection_name']}")
    print(f"  Chunks stored : {stats['total_chunks']}")