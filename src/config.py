import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

    # Paths
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    REPOS_PATH: str = os.getenv("REPOS_PATH", "./data/repos")

    # Embedding model (free, runs locally)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # LLM (via Groq, free)
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.0

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3

    # Supported file extensions
    SUPPORTED_EXTENSIONS: list = [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".cs", ".rb", ".php", ".swift", ".kt"
    ]

config = Config()