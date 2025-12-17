import asyncio
import os
from dotenv import load_dotenv

from src.mcp.mcp_client import MCPHttpSSEClient


load_dotenv()


async def main():
    client = MCPHttpSSEClient(
        base_url="http://localhost:8000",
        token=os.environ["MCP_TOKEN"],
    )

    try:
        print("[1] Connecting to MCP...")
        await asyncio.wait_for(client.connect(), timeout=10)
        print(f"[1] Connected. session_id={getattr(client, 'session_id', None)}")

        print("[2] Listing tools...")
        tools = await asyncio.wait_for(client.list_tools(), timeout=10)
        print("[2] TOOLS:", tools)

        print("[3] Calling tool: echo ...")
        res = await asyncio.wait_for(
            client.call_tool("echo", {"text": "ciao", "uppercase": True}),
            timeout=10,
        )
        print("[3] ECHO:", res)

    except asyncio.TimeoutError:
        print("\n[TIMEOUT] Operazione in timeout.")
        print("Suggerimento: il server potrebbe non inviare la risposta su SSE, oppure il metodo RPC non è corretto.")
        print(f"session_id={getattr(client, 'session_id', None)}")

    except Exception as e:
        print("\n[ERROR]", repr(e))
        print(f"session_id={getattr(client, 'session_id', None)}")

    finally:
        print("[4] Closing client...")
        try:
            await asyncio.wait_for(client.close(), timeout=5)
        except Exception as e:
            print("[WARN] close failed:", repr(e))
        print("[DONE]")


if __name__ == "__main__":
    print("RUNNING:")
    asyncio.run(main())