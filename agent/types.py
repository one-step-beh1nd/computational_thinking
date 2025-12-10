from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """Lightweight message container compatible with OpenAI chat API."""

    role: Role
    content: str | None = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_openai(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload

    @classmethod
    def from_openai(cls, message: Dict[str, Any]) -> "Message":
        """Create from OpenAI chat completion message payload."""
        return cls(
            role=message.get("role", "assistant"),
            content=message.get("content"),
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
            tool_calls=message.get("tool_calls"),
        )

    @staticmethod
    def format_tool_result(tool_name: str, tool_call_id: str, result: Any) -> "Message":
        """Normalize tool output into a message for LLM consumption."""
        if isinstance(result, str):
            content = result
        else:
            content = json.dumps(result, ensure_ascii=False)
        return Message(
            role="tool",
            name=tool_name,
            tool_call_id=tool_call_id,
            content=content,
        )

