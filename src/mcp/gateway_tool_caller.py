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
