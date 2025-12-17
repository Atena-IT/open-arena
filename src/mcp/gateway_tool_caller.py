from __future__ import annotations
from dotenv import load_dotenv
import requests
import asyncio
from typing import Any, Dict, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient


""" CONFIG """
load_dotenv()


""" CLASSES """
class GatewayToolCaller:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-MCP-Token": token, "Content-Type": "application/json"}

    def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "echo":
            r = requests.post(f"{self.base_url}/echo", json=args, headers=self.headers, timeout=300)
            r.raise_for_status()
            return r.json()

        if tool_name == "add":
            r = requests.post(f"{self.base_url}/add", json=args, headers=self.headers, timeout=300)
            r.raise_for_status()
            return r.json()

        if tool_name == "get_unix_time":
            r = requests.get(f"{self.base_url}/time", headers=self.headers, timeout=300)
            r.raise_for_status()
            return r.json()

        if tool_name == "simulate_kpi_fetch":
            r = requests.post(f"{self.base_url}/kpi", json=args, headers=self.headers, timeout=300)
            r.raise_for_status()
            return r.json()

        raise ValueError(f"Unknown tool: {tool_name}")



class MCPToolCaller:
    """
    Drop-in replacement di GatewayToolCaller, ma usando MCP (via MultiServerMCPClient).

    Uso async (consigliato):
        caller = MCPToolCaller(base_url="http://127.0.0.1:8000", headers={"X-MCP-Token": "..."})
        await caller.setup()
        res = await caller.acall("echo", {"text": "ciao", "uppercase": True})

    Uso sync:
        caller = MCPToolCaller(...)
        caller.setup_sync()
        res = caller.call("add", {"a": 1, "b": 2})
    """

    def __init__(self, base_url: str, *, server_name: str = "Gateway", mcp_path: str = "/mcp", transport: str = "sse", headers: Optional[Dict[str, str]] = None, timeout_s: Optional[float] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.server_name = server_name
        self.mcp_url = f"{self.base_url}{mcp_path}"
        self.transport = transport
        self.headers = headers or {}
        self.timeout_s = timeout_s  # non tutte le versioni lo usano direttamente

        self._client: Optional[MultiServerMCPClient] = None
        self._tools_by_name: Dict[str, Any] = {}
        self._setup_done: bool = False

    async def setup(self) -> None:
        """Inizializza il client MCP e scarica/cacha i tool remoti."""
        mcp_config = {
            self.server_name: {
                "transport": self.transport,
                "url": self.mcp_url,
                "headers": self.headers,
            }
        }

        self._client = MultiServerMCPClient(mcp_config)

        remote_tools = await self._client.get_tools()

        # Cache per lookup rapido per nome tool
        self._tools_by_name = {t.name: t for t in remote_tools}
        self._setup_done = True

    def setup_sync(self) -> None:
        """Wrapper sync per setup()."""
        asyncio.run(self.setup())

    async def acall(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Chiama un tool MCP per nome, passando gli argomenti.
        Ritorna l'output del tool (dipende dal server/tool).
        """
        if not self._setup_done:
            raise RuntimeError("MCPToolCaller non inizializzato. Chiama setup() prima.")

        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._tools_by_name.keys()))
            raise ValueError(f"Unknown tool: {tool_name}. Available: [{available}]")

        # Nei tool LangChain l'invocazione standard è invoke/ainvoke.
        # Usiamo ainvoke per rimanere async end-to-end.
        return await tool.ainvoke(args)

    def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Wrapper sync per acall()."""
        return asyncio.run(self.acall(tool_name, args))

    def list_tools(self) -> Dict[str, Any]:
        """Ritorna i tool disponibili (cache)."""
        if not self._setup_done:
            raise RuntimeError("MCPToolCaller non inizializzato. Chiama setup() prima.")
        return dict(self._tools_by_name)
