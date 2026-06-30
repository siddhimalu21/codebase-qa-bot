from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message in the conversation."""
    role: str       # 'user' or 'assistant'
    content: str


@dataclass
class ConversationMemory:
    """
    Stores conversation history and provides context for follow-up questions.
    Keeps last N turns to avoid exceeding token limits.
    """
    max_turns: int = 5
    messages: List[Message] = field(default_factory=list)

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str):
        """Add an assistant message to history."""
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def _trim(self):
        """Keep only the last max_turns * 2 messages."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history_text(self) -> str:
        """Format conversation history as text for the prompt."""
        if not self.messages:
            return ""

        lines = ["Previous conversation:"]
        for msg in self.messages[:-1]:  # exclude the latest user message
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content[:300]}")

        return "\n".join(lines)

    def is_followup(self) -> bool:
        """Return True if this is a follow-up question."""
        return len(self.messages) > 1

    def clear(self):
        """Clear conversation history."""
        self.messages = []

    def get_groq_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """
        Format messages for Groq API including history.
        Returns list of role/content dicts.
        """
        groq_messages = [{"role": "system", "content": system_prompt}]

        for msg in self.messages:
            groq_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        return groq_messages