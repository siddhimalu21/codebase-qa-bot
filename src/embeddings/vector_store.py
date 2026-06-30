import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from src.config import config
from src.chunking.models import CodeChunk


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=config.CHROMA_DB_PATH)


def get_or_create_collection(
    client: chromadb.PersistentClient,
    repo_name: str,
) -> chromadb.Collection:
    """Get existing collection or create a new one for a repo."""
    collection_name = f"repo_{repo_name}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def store_chunks(
    chunks: List[CodeChunk],
    embeddings: List[List[float]],
    repo_name: str,
) -> None:
    """
    Store chunks and their embeddings in ChromaDB.
    Handles batching automatically for large repos.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, repo_name)

    # Check how many are already stored
    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} chunks — clearing and re-indexing.")
        client.delete_collection(f"repo_{repo_name}")
        collection = get_or_create_collection(client, repo_name)

    print(f"Storing {len(chunks)} chunks in ChromaDB...")

    # ChromaDB has a limit of 5461 items per batch
    batch_size = 500
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i: i + batch_size]
        batch_embeddings = embeddings[i: i + batch_size]

        ids = [c.chunk_id for c in batch_chunks]
        documents = [c.content for c in batch_chunks]
        metadatas = [c.to_metadata() for c in batch_chunks]

        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total_stored += len(batch_chunks)
        print(f"  Stored {total_stored}/{len(chunks)} chunks...")

    print(f"Successfully stored {total_stored} chunks in ChromaDB.")


def query_collection(
    query_embedding: List[float],
    repo_name: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Search ChromaDB for the most similar chunks to a query.

    Returns:
        ChromaDB results dict with ids, documents, metadatas, distances
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client, repo_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return results


def get_collection_stats(repo_name: str) -> Dict[str, Any]:
    """Return stats about a stored collection."""
    client = get_chroma_client()
    collection = get_or_create_collection(client, repo_name)
    count = collection.count()
    return {
        "repo_name": repo_name,
        "collection_name": f"repo_{repo_name}",
        "total_chunks": count,
    }