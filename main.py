from src.ingestion import load_files_from_repo
from src.chunking import chunk_all_files
from src.embeddings import build_index
from src.llm import QAEngine
from src.evaluation import (
    run_evaluation,
    print_evaluation_report,
    FLASK_TEST_DATASET,
)


def main():
    print("=" * 40)
    print("   Codebase Q&A Bot — Day 9 Test")
    print("=" * 40)

    TEST_REPO = "https://github.com/pallets/flask"
    REPO_NAME = "flask"

    # Step 1 — Ingest
    print(f"\nStep 1: Ingesting repo...")
    files = load_files_from_repo(TEST_REPO)
    print(f"Loaded {len(files)} files")

    # Step 2 — Chunk
    print(f"\nStep 2: Chunking files...")
    chunks = chunk_all_files(files)
    print(f"Created {len(chunks)} chunks")

    # Step 3 — Build index
    print(f"\nStep 3: Building index...")
    build_index(chunks, REPO_NAME)

    # Step 4 — Initialize QA Engine
    print(f"\nStep 4: Initializing QA Engine...")
    engine = QAEngine(repo_name=REPO_NAME, all_chunks=chunks)

    # Step 5 — Run evaluation
    print(f"\nStep 5: Running RAGAS evaluation...")
    print(f"Evaluating on {len(FLASK_TEST_DATASET)} test questions...")
    print("-" * 40)

    scores = run_evaluation(engine, FLASK_TEST_DATASET)
    print_evaluation_report(scores)


if __name__ == "__main__":
    main()