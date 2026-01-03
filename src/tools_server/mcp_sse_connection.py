from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from litellm import experimental_mcp_client


""" CONFIG """
load_dotenv()


""" CLASS """
class MCPSSEConnection:
    def __init__(self, mcp_url: str, token: str):
        self.mcp_url = mcp_url
        self.headers = {"X-MCP-Token": token.strip()} if token else {}

    async def __aenter__(self):

        # SSE Configuration
        self._sse_cm = sse_client(
            url=self.mcp_url,
            headers=self.headers,
            timeout=60,
            sse_read_timeout=60,
        )

        # Opening and initialize SSE connection with channels read & write
        self._read_write = await self._sse_cm.__aenter__()
        read, write = self._read_write

        # Opening and initialize MCP connection
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()

        # Loading tools and converting them in OpenAI format
        self.tools = await experimental_mcp_client.load_mcp_tools(
            session=self.session,
            format="openai",
        )
        return self

    # Closing MCP and SSE connections
    async def __aexit__(self, exc_type, exc, tb):
        await self.session.__aexit__(exc_type, exc, tb)
        await self._sse_cm.__aexit__(exc_type, exc, tb)
