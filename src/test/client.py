import os, asyncio
from dotenv import load_dotenv
from src.test.llm_client import LLMClient
from src.mcp_server.mcp_sse_bearer import MCPSSEBearer


""" CONFIG """
load_dotenv()
TOKEN = os.getenv("MCP_TOKEN", "")
URL =  f'{str(os.getenv("FAST_API_URL", ""))}:{str(os.getenv("FAST_API_PORT", ""))}/{str(os.getenv("MCP_PATH", ""))}'


""" FUNCTIONS """
async def main():
    llm = LLMClient()
    messages = llm.format_messages(
        system="Use available tools when needed.",
        user="Retrieve the billing success_rate in US and add 2497 to 3843. After repeat 'ok' in uppercase."
    )
    model_config = {
        "name": "gpt-4o-mini",
        "max_tokens": 500,
        "temperature": 0.0,
    }

    async with MCPSSEBearer(mcp_url=URL, token=TOKEN) as mcp:
        print(await llm.chat_with_mcp_tools(
            messages=messages,
            model_config=model_config,
            mcp_session=mcp.session,
            mcp_tools_openai=mcp.tools,
        ))


""" MAIN """
if __name__ == "__main__":
    asyncio.run(main())
