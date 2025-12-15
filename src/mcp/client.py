import asyncio
from fastmcp import FastMCPClient

async def main():
    # Start your MCP server as a subprocess
    client = await FastMCPClient.create(
        cmd=["python", "-m", "src.mcp.server"],  # adjust module path
        name="MultiModelEvalServer",            # must match FastMCP("MultiModelEvalServer")
    )

    # List available tools
    tools = await client.list_tools()
    print("Available tools:", [t.name for t in tools])

    # Call QA pipeline tool
    qa_result = await client.call_tool("run_qa_pipeline")
    print("QA pipeline result:", qa_result)

    # Call ToolScale pipeline tool
    toolscale_result = await client.call_tool("run_toolscale_pipeline")
    print("ToolScale pipeline result:", toolscale_result)

    # Call model listing tool
    models = await client.call_tool("list_models_under_test")
    print("Models under test:", models)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())