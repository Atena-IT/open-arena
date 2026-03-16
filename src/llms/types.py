from typing import TypedDict, NotRequired

class MCPServerConfig(TypedDict):
    """
    Configuration for an MCP (Model Context Protocol) server.

    Required fields:
        server_name: Name identifier for the server
        url: SSE endpoint URL for the remote MCP server

    Optional fields:
        headers: HTTP headers for authentication (e.g., Authorization bearer token)
    """
    server_name: str
    url: str
    headers: NotRequired[dict[str, str]]
