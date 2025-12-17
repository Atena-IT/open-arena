import os
from dotenv import load_dotenv
from src.mcp.gateway_tool_caller import GatewayToolCaller
from src.mcp.llm_client import LLMClient


""" CONFIG """
GATEWAY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Returns the input text unchanged, or uppercased if uppercase=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "uppercase": {"type": "boolean", "default": False},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Returns the sum of two given float numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unix_time",
            "description": "Returns the unix time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_kpi_fetch",
            "description": "Returns a mocked KPI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system": {"type": "string"},
                    "kpi_name": {"type": "string"},
                    "region": {"type": "string", "default": "eu"},
                },
                "required": ["system", "kpi_name"],
            },
        },
    },
]
load_dotenv()


""" MAIN """
if __name__ == "__main__":
    client = LLMClient()
    gateway = GatewayToolCaller(base_url="http://localhost:8000", token=os.getenv("MCP_TOKEN", ""))

    messages = client.format_messages(
        system="Usa i tool disponibili quando servono.",
        user="Recupera il success_rate di billing in US e poi somma 24 + 38. Infine ripeti 'ok' in maiuscolo."
    )

    model_config = {
        "name": "gpt-4o-mini",
        "max_tokens": 500,
        "temperature": 0.0,
        "tools": GATEWAY_TOOLS,
        "stream": False,
    }

    print(client.chat_with_tools(messages, model_config, gateway))