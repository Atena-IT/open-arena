import asyncio
import json
import os
from typing import Any, Dict, Optional

import httpx


class MCPHttpSSEClient:
    """
    Minimal MCP-over-SSE client for a FastAPI MCP endpoint mounted at /mcp.

    It:
      - obtains/keeps a session id
      - opens SSE stream (GET /mcp)
      - sends JSON-RPC messages (POST /mcp)
      - receives responses on SSE stream
    """

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

        self.session_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_cm = None
        self._stream_resp: Optional[httpx.Response] = None

        self._next_id = 1
        self._pending: Dict[str, asyncio.Future] = {}

    def _headers(self, *, accept_sse: bool = False) -> Dict[str, str]:
        h = {
            "X-MCP-Token": self.token,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        if accept_sse:
            h["Accept"] = "text/event-stream"
        if self.session_id:
            # IMPORTANT: name is exactly what the server returned
            h["mcp-session-id"] = self.session_id
        return h

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    async def _bootstrap_session(self) -> None:
        """
        First touch /mcp to get a session id (server returns it in response headers).
        Even if the response is 400, we still capture mcp-session-id.
        """
        client = await self._ensure_client()
        r = await client.get(f"{self.base_url}/mcp", headers=self._headers(accept_sse=True))
        sid = r.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid

        # After capturing session id, we don't care if this first call failed
        # because server explicitly said session id was missing.
        # We'll retry with session id in connect().
        return

    '''
    async def connect(self) -> None:
        """
        Opens the SSE stream and starts background task to read incoming messages.
        """
        await self._bootstrap_session()
        if not self.session_id:
            raise RuntimeError("Could not obtain mcp-session-id from server headers")

        client = await self._ensure_client()
        self._stream_cm = client.stream(
            "GET",
            f"{self.base_url}/mcp",
            headers=self._headers(accept_sse=True),
        )
        self._stream_resp = await self._stream_cm.__aenter__()
        if self._stream_resp.status_code >= 400:
            body = await self._stream_resp.aread()
            raise RuntimeError(
                f"SSE connect failed: {self._stream_resp.status_code} {body.decode('utf-8', 'replace')}"
            )

        # Start reader loop
        asyncio.create_task(self._reader_loop())

        # Initialize MCP session (JSON-RPC initialize is usually required)
        # If your server requires different method names, the error response will tell us.
        await self.rpc("initialize", {"clientInfo": {"name": "litellm-bridge", "version": "0.1"}})
    '''

    async def connect(self) -> None:
        await self._bootstrap_session()
        if not self.session_id:
            raise RuntimeError("Could not obtain mcp-session-id from server headers")

        client = await self._ensure_client()
        self._stream_cm = client.stream(
            "GET",
            f"{self.base_url}/mcp",
            headers=self._headers(accept_sse=True),
        )
        self._stream_resp = await self._stream_cm.__aenter__()
        if self._stream_resp.status_code >= 400:
            body = await self._stream_resp.aread()
            raise RuntimeError(
                f"SSE connect failed: {self._stream_resp.status_code} {body.decode('utf-8', 'replace')}"
            )

        asyncio.create_task(self._reader_loop())

        # TEMP: non chiamare initialize finché non sappiamo il method name corretto
        return



    async def close(self) -> None:
        if self._stream_cm is not None:
            await self._stream_cm.__aexit__(None, None, None)
            self._stream_cm = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _reader_loop(self) -> None:
        """
        Reads SSE stream and resolves pending JSON-RPC futures.

        SSE events may contain multiple `data:` lines. An event ends with an empty line.
        We concatenate all data lines and parse JSON once per event.
        """
        assert self._stream_resp is not None

        data_lines = []

        async for line in self._stream_resp.aiter_lines():
            if line is None:
                continue

            # Event boundary (empty line) => process collected data lines
            if line == "":
                if not data_lines:
                    continue

                raw = "\n".join(data_lines).strip()
                data_lines = []

                # Some servers send non-JSON keepalive messages; ignore parse errors
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is None:
                    continue

                fut = self._pending.pop(str(msg_id), None)
                if fut and not fut.done():
                    fut.set_result(msg)

                continue

            # Collect data lines
            if line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
                continue

            # Optional: ignore other SSE fields like 'event:', 'id:', 'retry:'
            continue

    async def rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session_id:
            raise RuntimeError("Not connected (missing session id). Call connect() first.")

        client = await self._ensure_client()

        req_id = str(self._next_id)
        self._next_id += 1

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[req_id] = fut

        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}

        r = await client.post(
            f"{self.base_url}/mcp",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
        )

        # If server replies inline, use it (even for errors)
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            msg = r.json()
            if "error" in msg:
                raise RuntimeError(f"RPC error for {method}: {msg['error']}")
            if "result" in msg or msg.get("id") == req_id:
                return msg

        # If POST returned an error but not JSON, print body
        if r.status_code >= 400:
            body = await r.aread()
            raise RuntimeError(
                f"POST /mcp failed ({r.status_code}, {ctype}): {body.decode('utf-8', 'replace')}"
            )

        # Otherwise wait for SSE response
        try:
            msg = await asyncio.wait_for(fut, timeout=30)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout waiting SSE response for method={method}, id={req_id}")
        finally:
            self._pending.pop(req_id, None)

        if "error" in msg:
            raise RuntimeError(f"RPC error for {method}: {msg['error']}")

        return msg

    async def list_tools(self) -> Dict[str, Any]:
        for m in ("tools/list", "tools.list", "list_tools", "tools"):
            try:
                return await self.rpc(m, {})
            except RuntimeError as e:
                last = e
        raise last  # type: ignore

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        for m in ("tools/call", "tools.call", "call_tool"):
            try:
                return await self.rpc(m, {"name": name, "arguments": arguments})
            except RuntimeError as e:
                last = e
        raise last  # type: ignore