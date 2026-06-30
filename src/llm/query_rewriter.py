from src.llm.groq_client import get_groq_client
from src.config import config


REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a codebase Q&A system.

Your job: given a conversation history and a follow-up question, 
rewrite the follow-up question to be fully self-contained.

Rules:
- Replace pronouns like "it", "this", "that" with the actual thing being referred to
- Keep the rewritten query short and specific
- If the question is already self-contained, return it unchanged
- Return ONLY the rewritten query, nothing else

Example:
History: User asked about the login_required decorator
Follow-up: "where is it used?"
Rewritten: "where is login_required decorator used in the codebase?"
"""


def rewrite_query(query: str, history_text: str) -> str:
    """
    Rewrite a follow-up query to be self-contained using conversation history.

    Args:
        query: The follow-up question
        history_text: Formatted conversation history

    Returns:
        Rewritten query string
    """
    if not history_text:
        return query

    client = get_groq_client()

    prompt = f"""{history_text}

Follow-up question: {query}

Rewrite the follow-up question to be self-contained:"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten
    except Exception:
        # If rewriting fails, return original query
        return query