from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .types import Message


class LLMClient:
    """
    Thin async wrapper around the OpenAI-compatible chat-completions API.

    The model ID and base URL are intentionally left blank: set them via
    environment variables before running:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL   (e.g. https://api.openai.com/v1 or your proxy)
      - OPENAI_MODEL      (e.g. gpt-4o-mini)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "")
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", None),
        )

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call chat completions and return the raw message dict."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[m.to_openai() for m in messages],
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        # Return the first choice's message payload for simplicity
        return resp.choices[0].message.model_dump()

