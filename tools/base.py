from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render_for_llm(self) -> str:
        """Return a compact string safe for feeding back to the LLM."""
        return json.dumps(
            {
                "success": self.success,
                "output": self.output,
                "error": self.error,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )


class BaseTool(ABC):
    """Abstract tool interface."""

    name: str
    description: str

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters (OpenAI function format)."""
        return {"type": "object", "properties": {}, "required": []}

    def to_function(self) -> Dict[str, Any]:
        """OpenAI tool spec wrapper."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def __call__(self, **kwargs: Any) -> ToolResult:
        ...

