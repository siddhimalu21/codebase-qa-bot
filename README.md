# 🔍 Codebase Q&A Bot — Retrieval-Augmented GitHub Repository Assistant

> Ask natural language questions about any GitHub codebase and get precise answers with exact file and line citations.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-pink?style=flat-square)

---

## 🚀 What it does

Codebase Q&A Bot enables developers to explore and understand any public GitHub repository using natural language. It indexes the repository, retrieves the most relevant code using a hybrid retrieval pipeline, and generates grounded answers with precise file paths and line-number citations.

Example questions:

- How does Flask handle URL routing?
- Where is `login_required` decorator defined?
- Explain the execution flow from `uvicorn.run()` to endpoint response.

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/Codebase Q&A Bot.png" alt="Codebase Q&A Bot" width="400"/>
</p>

---
## 🎯 Why this project?

Understanding large codebases can be time-consuming, and traditional code search often relies on exact keyword matches. This project combines AST-aware chunking, hybrid retrieval 
(semantic + keyword search), cross-encoder re-ranking, and an LLM to provide accurate, context-aware answers with precise file and line citations. 
The goal is to make navigating and understanding unfamiliar GitHub repositories faster and more reliable.

---

## ✨ Key Features

### AST-Aware Chunking
Uses Python's ast module to split code at function and class boundaries. Each chunk = one complete function with its signature and docstring, preserving semantic meaning for much better retrieval.

### Hybrid Search
Combines dense vector search (semantic similarity) with BM25 keyword search (exact matches). Results are fused using Reciprocal Rank Fusion (RRF).

### Cross-Encoder Re-ranking
After retrieving top-10 chunks, a cross-encoder model re-scores each query-chunk pair jointly. Top-3 re-ranked chunks are passed to the LLM.

### Conversation Memory
Multi-turn chat with automatic query rewriting. Follow-up questions like "where is it used?" are automatically expanded using conversation history.

---

## 📊 Evaluation Results

The system was evaluated on **5 question-answer pairs** from the Flask codebase using an **LLM-as-a-Judge** framework.

| Metric | Score | Description |
|---------|:-----:|-------------|
| Faithfulness | **0.72** | Measures whether the generated answer is supported by the retrieved code context. |
| Answer Relevancy | **0.88** | Measures how well the answer addresses the user's question. |
| Context Precision | **0.88** | Measures the relevance of retrieved code chunks used for answer generation. |
| **Overall Average** | **0.83** | Average score across all evaluation metrics. |

> **Evaluation Setup**
> - **Dataset:** 5 manually curated questions on the Flask codebase
> - **Evaluation Method:** LLM-as-a-Judge
> - **LLM:** Groq LLaMA 3.1 8B

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Programming Language** | Python 3.11 | Core application development |
| **LLM** | Groq LLaMA 3.1 8B | Natural language question answering |
| **Framework** | LangChain 0.3 | RAG pipeline orchestration |
| **Repository Ingestion** | GitPython | Clone and process GitHub repositories |
| **Code Parsing** | Python AST | Function and class-level code chunking |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic code embeddings |
| **Vector Database** | ChromaDB | Persistent vector storage and similarity search |
| **Keyword Search** | rank-bm25 | Lexical retrieval using BM25 |
| **Conversation Memory** | LangChain Memory | Maintains multi-turn conversation context |
| **User Interface** | Streamlit | Interactive web application |
---

## 🏁 Getting Started

### Prerequisites
- Python 3.11+
- Git
- Groq API key (free at console.groq.com)

### Installation

Clone the repo:
```bash
git clone https://github.com/siddhimalu21/codebase-qa-bot.git
cd codebase-qa-bot
```

Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Create a .env file:
```env
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./data/chroma_db
REPOS_PATH=./data/repos
```

### Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501, paste any public GitHub repository URL, wait for indexing to complete and then ask natural-language questions about the codebase.

---

## 📂 Project Structure

```text
codebase-qa-bot/
├── src/
│   ├── ingestion/
│   │   ├── models.py
│   │   ├── repo_loader.py
│   │   └── language_detector.py
│   ├── chunking/
│   │   ├── models.py
│   │   ├── python_chunker.py
│   │   ├── generic_chunker.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── vector_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── llm/
│   │   ├── prompt_builder.py
│   │   ├── groq_client.py
│   │   ├── qa_engine.py
│   │   ├── memory.py
│   │   └── query_rewriter.py
│   └── evaluation/
│       ├── evaluator.py
│       └── test_dataset.py
├── app.py
├── main.py
└── requirements.txt
```

---


## 💬 Example Queries

After indexing the **Flask** repository:

> Repository: https://github.com/pallets/flask

### 1. How does Flask handle URL routing?

**Retrieved from:**

```text
src/flask/sansio/scaffold.py
Lines 336–365
```

---

### 2. Where is `login_required` defined?

**Retrieved from:**

```text
examples/tutorial/flaskr/auth.py
Lines 19–29
```

---

### 3. How do blueprints work?

**Retrieved from:**

```text
src/flask/sansio/blueprints.py
Lines 119–692
```

---

## 🔮 Future Improvements

- Support AST-based chunking for JavaScript and TypeScript
- Multi-repository indexing and querying
- Dockerized deployment
- GitHub Actions CI/CD pipeline
- Repository-level caching for faster indexing
- Tree-sitter support for additional programming languages

---

## Author

Siddhi Malu
GitHub: [@siddhimalu21](https://github.com/siddhimalu21)

---

## License

MIT License - feel free to use this project as a reference or build on top of it.