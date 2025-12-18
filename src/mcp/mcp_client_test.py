import os, asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

async def main():
    # 1) MCP client verso il tuo server
    client = MultiServerMCPClient(
        {
            "Demo-MCP": {
                "transport": "sse",
                "url": "http://127.0.0.1:8000/mcp",
                "headers": {
                    # deve combaciare con il tuo Header alias="X-MCP-Token"
                    "X-MCP-Token": os.getenv("MCP_TOKEN", "").strip()
                },
            }
        }
    )

    remote_tools = await client.get_tools()
    print("TOOLS:", [t.name for t in remote_tools])

    # 2) Modello + tool binding
    model = init_chat_model("openai:gpt-4o-mini")

    def call_model(state: MessagesState):
        resp = model.bind_tools(remote_tools).invoke(state["messages"])
        return {"messages": resp}

    # 3) Graph che gestisce automaticamente tool call/response
    g = StateGraph(MessagesState)
    g.add_node("call_model", call_model)
    g.add_node("tools", ToolNode(remote_tools))
    g.add_edge(START, "call_model")
    g.add_conditional_edges("call_model", tools_condition)
    g.add_edge("tools", "call_model")
    graph = g.compile()

    # 4) Run
    messages = [
        SystemMessage(content="Usa i tool MCP disponibili quando servono."),
        HumanMessage(content="Recupera il success_rate di billing in US e poi somma 2497 + 3843. Infine ripeti 'ok' in maiuscolo.")
    ]

    out = await graph.ainvoke({"messages": messages})
    print("\nFINAL:\n", out["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())