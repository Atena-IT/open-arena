import os, asyncio
from dotenv import load_dotenv
from src.mcp.llm_client import LLMClient
from src.mcp.mcp_sse_tooling import MCPSSETooling


""" CONFIG """
load_dotenv()


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

    async with MCPSSETooling(mcp_url="http://127.0.0.1:8000/mcp", token=os.getenv("MCP_TOKEN", "")) as mcp:
        print(await llm.chat_with_mcp_tools(
            messages=messages,
            model_config=model_config,
            mcp_session=mcp.session,
            mcp_tools_openai=mcp.tools,
        ))


""" MAIN """
if __name__ == "__main__":
    asyncio.run(main())
