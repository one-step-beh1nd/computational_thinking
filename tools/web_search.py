from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """
    Web search via Serper API.

    Setup:
      - Register for a key at https://serper.dev
      - Export SERPER_API_KEY=your_key
    """

    name = "web_search"
    description = "Search the web and return top snippets."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many results to return (<=10 recommended)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def __call__(self, query: str, top_k: int = 5, **_: Any) -> ToolResult:
        load_dotenv()
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="SERPER_API_KEY missing. Get one at https://serper.dev and export it.",
            )

        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": max(1, min(top_k, 10))}

        resp = requests.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            return ToolResult(success=False, error=f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        items: List[Dict[str, Any]] = []
        for block in ("organic", "news"):
            for item in data.get(block, []):
                items.append(
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": block,
                    }
                )

        return ToolResult(success=True, output=items[:top_k], metadata={"provider": "serper"})

