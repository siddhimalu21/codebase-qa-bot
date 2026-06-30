from typing import List, Dict
from groq import Groq
from src.config import config


def get_groq_client() -> Groq:
    """Return a configured Groq client."""
    return Groq(api_key=config.GROQ_API_KEY)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call Groq's Llama 3.1 model and return the response text.
    """
    client = get_groq_client()

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def call_llm_with_history(messages: List[Dict[str, str]]) -> str:
    """
    Call Groq with full conversation history.

    Args:
        messages: List of role/content dicts including system prompt

    Returns:
        LLM response as string
    """
    client = get_groq_client()

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def call_llm_stream(system_prompt: str, user_prompt: str):
    """
    Stream the LLM response token by token.
    Used in the Streamlit UI for real-time output.
    """
    client = get_groq_client()

    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def call_llm_stream_with_history(messages: List[Dict[str, str]]):
    """
    Stream LLM response with full conversation history.
    """
    client = get_groq_client()

    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta