from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from litellm import experimental_mcp_client


""" CONFIG """
load_dotenv()


""" CLASS """
class MCPSSEBearer:
    """
    Async context manager for establishing an authenticated MCP session over SSE.
    Parameters:
        :param mcp_url: Server access url for MCP.
        :param token: MCP access token.
    """
    def __init__(self, mcp_url: str, token: str):
        self.mcp_url = mcp_url
        self.headers = {"X-MCP-Token": token.strip()} if token else {}
        self.sse_cm = sse_client(
            url=self.mcp_url,
            headers=self.headers,
            timeout=60,
            sse_read_timeout=60,
        )


    async def __aenter__(self):
        """
        Opens the SSE transport (read/write channels), creates and initializes an MCP 'ClientSession', then loads the
        available MCP tools and converts them to the OpenAI tool schema
        """
        self._read_write = await self.sse_cm.__aenter__()
        read, write = self._read_write

        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()

        self.tools = await experimental_mcp_client.load_mcp_tools(
            session=self.session,
            format="openai",
        )
        return self


    async def __aexit__(self, exc_type, exc, tb):
        """
        Closes MCP session and SSE connection
        """
        await self.session.__aexit__(exc_type, exc, tb)
        await self.sse_cm.__aexit__(exc_type, exc, tb)
