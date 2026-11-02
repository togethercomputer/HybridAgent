"""Tool registry and executor for the hybrid generation engine, plus the default tool set
of HybridAgent: a Python interpreter tool and a live search tool.
"""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from typing import Dict, Optional

from .base import BaseTool
from .python_tool import PythonTool
from .search_tool import TavilySearchTool
from .youcom_search_tool import YouComSearchTool

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Registry for tools; handles identification, extraction, and execution."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.trigger_tag] = tool

    def get_tool(self, tag: str) -> Optional[BaseTool]:
        return self.tools.get(tag)

    def execute(self, tag: str, content: str, **kwargs) -> str:
        tool = self.get_tool(tag)
        if not tool:
            raise ValueError(f"Tool '{tag}' is not registered")
        try:
            return tool.execute(content, **kwargs)
        except Exception as e:
            logger.error(f"Tool({tag}) execution failed: {e}", exc_info=True)
            return f"Tool execution failed: {e}"

    def identify_tool(self, text: str) -> Optional[str]:
        """Return the trigger_tag if the text contains a closing tool tag."""
        for tag in self.tools:
            if f"</{tag}>" in text:
                return tag
        return None

    @staticmethod
    def extract_content(text: str, tag: str) -> str:
        """Extract text between the *last* <tag>...</tag> pair."""
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        try:
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag) : end_pos].strip()
        except ValueError:
            return ""


class RetryingSearchTool:
    """Wraps a search tool with request jitter and extra retries for batched rollouts."""

    def __init__(self, inner: BaseTool):
        self._inner = inner
        self.calls = 0
        self.failures = 0

    @property
    def name(self):
        return self._inner.name

    @property
    def trigger_tag(self):
        return self._inner.trigger_tag

    def execute(self, content: str, **kwargs) -> str:
        self.calls += 1
        time.sleep(random.uniform(0.0, 0.75))
        out = self._inner.execute(content, **kwargs)
        if isinstance(out, str) and out.startswith("Search failed after"):
            self.failures += 1
            for attempt in range(3):
                time.sleep(random.uniform(1.0, 2.0) * (2**attempt))
                out = self._inner.execute(content, **kwargs)
                if not (isinstance(out, str) and out.startswith("Search failed after")):
                    self.failures -= 1
                    break
        return out


def build_tool_executor(
    python_timeout: int = 120,
    search_max_results: int = 10,
    search_result_length: int = 1000,
    *,
    require_search: bool = True,
    search_provider: str | None = None,
) -> ToolExecutor:
    """Python interpreter + live search.

    Search defaults to Tavily (TAVILY_API_KEY). ``search_provider="youcom"`` (or
    HYBRID_SEARCH_PROVIDER=youcom) switches to You.com, which reads YDC_API_KEY.
    """
    executor = ToolExecutor()
    executor.register_tool(PythonTool(python_path=sys.executable, timeout=python_timeout))
    provider = (search_provider or os.environ.get("HYBRID_SEARCH_PROVIDER") or "tavily").strip().lower()
    if provider not in ("tavily", "youcom"):
        raise ValueError(f"Unknown search provider {provider!r}; expected 'tavily' or 'youcom'")
    env_var = "YDC_API_KEY" if provider == "youcom" else "TAVILY_API_KEY"
    key = os.environ.get(env_var, "")
    if key:
        tool_cls = YouComSearchTool if provider == "youcom" else TavilySearchTool
        executor.register_tool(RetryingSearchTool(tool_cls(
            api_key=key,
            max_results=search_max_results,
            result_length=search_result_length,
            max_retries=3,
            retry_delay=2.0,
        )))
    elif require_search:
        raise RuntimeError(f"{env_var} is required for the search tool (set it in .env)")
    return executor
