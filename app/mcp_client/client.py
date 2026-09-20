"""MCP client used by the agent.

The agent does not import the tool functions. It discovers tools with `tools/list` and
invokes them with `tools/call` over the configured transport, exactly as an external MCP host
would. That is what makes the tool layer swappable and what lets the same server be used by
other MCP hosts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Optional

from mcp import Client, StdioServerParameters

from app.config import Settings

logger = logging.getLogger(__name__)


class MCPUnavailable(Exception):
    """The MCP server could not be reached or the session is not open."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def signature(self) -> str:
        props = (self.input_schema or {}).get("properties", {}) or {}
        required = set((self.input_schema or {}).get("required", []) or [])
        parts = []
        for key, spec in props.items():
            type_name = spec.get("type") or "any"
            if "anyOf" in spec:
                type_name = "/".join(
                    opt.get("type", "any") for opt in spec["anyOf"] if opt.get("type") != "null"
                ) or "any"
            parts.append(f"{key}: {type_name}{'' if key in required else '?'}")
        return f"{self.name}({', '.join(parts)})"


class MCPToolClient:
    """Long-lived MCP session with a uniform call interface and hard timeouts."""

    def __init__(self, settings: Settings, server_object: Any = None):
        self.settings = settings
        self.transport = settings.mcp_transport
        self._server_object = server_object
        self._client: Optional[Client] = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools: dict[str, ToolSpec] = {}
        self._lock = asyncio.Lock()

    # --- lifecycle ---------------------------------------------------------
    def _build_client(self) -> Client:
        if self.transport == "inproc":
            if self._server_object is None:
                raise MCPUnavailable("MCP_TRANSPORT=inproc requires an in-process server object")
            return Client(self._server_object)
        if self.transport == "stdio":
            env = dict(os.environ)
            return Client(StdioServerParameters(
                command=sys.executable,
                args=["-m", "app.mcp_server.server"],
                env=env,
            ))
        return Client(self.settings.mcp_server_url)

    async def connect(self) -> None:
        if self._client is not None:
            return
        stack = AsyncExitStack()
        client = self._build_client()
        try:
            self._client = await stack.enter_async_context(client)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as MCPUnavailable
            await stack.aclose()
            self._client = None
            raise MCPUnavailable(f"Could not open MCP session ({self.transport}): {exc}") from exc
        self._stack = stack
        await self.refresh_tools()
        logger.info("MCP session open (%s) with %d tools", self.transport, len(self._tools))

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                # Closing from a different task than the one that opened the session makes
                # anyio complain; the subprocess/stream is torn down either way.
                logger.debug("MCP session close reported: %s", exc)
        self._stack = None
        self._client = None
        self._tools = {}

    # --- discovery ---------------------------------------------------------
    async def refresh_tools(self) -> dict[str, ToolSpec]:
        if self._client is None:
            raise MCPUnavailable("MCP session is not open")
        result = await self._client.list_tools()
        raw_tools = getattr(result, "tools", result)
        specs: dict[str, ToolSpec] = {}
        for tool in raw_tools:
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
            specs[tool.name] = ToolSpec(
                name=tool.name,
                description=(tool.description or "").strip(),
                input_schema=schema if isinstance(schema, dict) else {},
            )
        self._tools = specs
        return specs

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return self._tools

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def describe_tools(self, names: Optional[list[str]] = None, max_chars: int = 420) -> str:
        """Compact catalogue for the planner prompt."""
        chosen = [self._tools[n] for n in (names or self.tool_names()) if n in self._tools]
        lines = []
        for spec in chosen:
            description = " ".join(spec.description.split())
            if len(description) > max_chars:
                description = description[: max_chars - 1] + "…"
            lines.append(f"- {spec.signature()}\n  {description}")
        return "\n".join(lines)

    # --- invocation --------------------------------------------------------
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool and always return a dict envelope (never raise on tool failure)."""
        if self._client is None:
            return {"ok": False, "tool": name,
                    "error": {"kind": "mcp_unavailable", "message": "MCP session is not open"},
                    "items": [], "count": 0}
        if name not in self._tools:
            return {"ok": False, "tool": name,
                    "error": {"kind": "unknown_tool",
                              "message": f"Unknown tool '{name}'. Available: {', '.join(self.tool_names())}"},
                    "items": [], "count": 0}
        cleaned = {k: v for k, v in (arguments or {}).items() if v not in (None, "", [], {})}
        try:
            async with self._lock:
                result = await asyncio.wait_for(
                    self._client.call_tool(name, cleaned),
                    timeout=self.settings.mcp_tool_timeout_seconds,
                )
        except asyncio.TimeoutError:
            return {"ok": False, "tool": name,
                    "error": {"kind": "timeout",
                              "message": f"Tool '{name}' exceeded {self.settings.mcp_tool_timeout_seconds}s"},
                    "items": [], "count": 0}
        except Exception as exc:  # noqa: BLE001 - transport failures are data for the agent
            return {"ok": False, "tool": name,
                    "error": {"kind": "transport_error", "message": f"{type(exc).__name__}: {exc}"},
                    "items": [], "count": 0}
        return self._unwrap(name, result)

    @staticmethod
    def _unwrap(name: str, result: Any) -> dict[str, Any]:
        is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
        structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
        if isinstance(structured, dict) and "ok" in structured:
            return structured

        texts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        joined = "\n".join(texts).strip()
        if joined:
            try:
                parsed = json.loads(joined)
                if isinstance(parsed, dict):
                    if is_error and "ok" not in parsed:
                        parsed["ok"] = False
                    return parsed
                return {"ok": not is_error, "tool": name, "items": [], "count": 0, "raw": parsed}
            except json.JSONDecodeError:
                pass
        if is_error:
            return {"ok": False, "tool": name,
                    "error": {"kind": "tool_error", "message": joined or "Tool reported an error"},
                    "items": [], "count": 0}
        if isinstance(structured, dict):
            return structured
        return {"ok": True, "tool": name, "items": [], "count": 0, "raw": joined}
