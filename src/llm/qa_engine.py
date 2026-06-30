from typing import List, Dict, Any
from src.chunking.models import CodeChunk
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import get_reranker
from src.llm.prompt_builder import build_prompt, SYSTEM_PROMPT
from src.llm.groq_client import call_llm_with_history, call_llm_stream_with_history
from src.llm.memory import ConversationMemory
from src.llm.query_rewriter import rewrite_query


class QAEngine:
    """
    End-to-end Q&A engine with re-ranking and conversation memory.

    Flow:
        query -> rewrite if follow-up
              -> hybrid search (top 10)
              -> re-ranker (top 3)
              -> prompt builder
              -> LLM with history
              -> answer with citations
    """

    def __init__(self, repo_name: str, all_chunks: List[CodeChunk]):
        self.repo_name = repo_name
        self.all_chunks = all_chunks
        self.memory = ConversationMemory(max_turns=5)
        print(f"QAEngine ready for repo: {repo_name}")
        print(f"Total chunks available: {len(all_chunks)}")

    def answer(
        self,
        query: str,
        retrieval_k: int = 10,
        rerank_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Answer a question with memory and re-ranking.
        """
        # Step 1 — rewrite follow-up queries
        history_text = self.memory.get_history_text()
        if self.memory.is_followup():
            rewritten_query = rewrite_query(query, history_text)
            print(f"Rewritten query: {rewritten_query}")
        else:
            rewritten_query = query

        # Step 2 — hybrid retrieval
        retrieved_chunks = hybrid_search(
            query=rewritten_query,
            repo_name=self.repo_name,
            all_chunks=self.all_chunks,
            top_k=retrieval_k,
        )

        if not retrieved_chunks:
            answer_text = "No relevant code found for this question."
            self.memory.add_user_message(query)
            self.memory.add_assistant_message(answer_text)
            return {
                "answer": answer_text,
                "sources": [],
                "chunks": [],
                "rewritten_query": rewritten_query,
            }

        # Step 3 — re-rank
        reranker = get_reranker()
        reranked_chunks = reranker.rerank_chunks_only(
            query=rewritten_query,
            chunks=retrieved_chunks,
            top_k=rerank_k,
        )

        # Step 4 — build prompt with context
        user_prompt = build_prompt(rewritten_query, reranked_chunks)

        # Step 5 — add to memory and call LLM with history
        self.memory.add_user_message(user_prompt)
        messages = self.memory.get_groq_messages(SYSTEM_PROMPT)
        answer_text = call_llm_with_history(messages)

        # Step 6 — store answer in memory
        self.memory.add_assistant_message(answer_text)

        # Step 7 — build sources
        sources = []
        for chunk in reranked_chunks:
            name = chunk.function_name or chunk.class_name or "block"
            sources.append({
                "file": chunk.file_path,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "name": name,
                "type": chunk.chunk_type,
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks": reranked_chunks,
            "rewritten_query": rewritten_query,
        }

    def answer_stream(
        self,
        query: str,
        retrieval_k: int = 10,
        rerank_k: int = 3,
    ):
        """Stream the answer token by token."""
        history_text = self.memory.get_history_text()
        if self.memory.is_followup():
            rewritten_query = rewrite_query(query, history_text)
        else:
            rewritten_query = query

        retrieved_chunks = hybrid_search(
            query=rewritten_query,
            repo_name=self.repo_name,
            all_chunks=self.all_chunks,
            top_k=retrieval_k,
        )

        if not retrieved_chunks:
            yield "No relevant code found for this question."
            return

        reranker = get_reranker()
        reranked_chunks = reranker.rerank_chunks_only(
            query=rewritten_query,
            chunks=retrieved_chunks,
            top_k=rerank_k,
        )

        user_prompt = build_prompt(rewritten_query, reranked_chunks)
        self.memory.add_user_message(user_prompt)
        messages = self.memory.get_groq_messages(SYSTEM_PROMPT)

        full_response = ""
        for token in call_llm_stream_with_history(messages):
            full_response += token
            yield token

        self.memory.add_assistant_message(full_response)

    def clear_memory(self):
        """Reset conversation history."""
        self.memory.clear()
        print("Conversation memory cleared.")