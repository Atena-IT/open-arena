import json, litellm, os
from dotenv import load_dotenv
from langfuse import Langfuse
from src.mcp.gateway_tool_caller import GatewayToolCaller


""" CONFIG """
load_dotenv()


""" CLASSES """
class LLMClient:
    """
    Client for interacting with the selected model using LiteLLM and Langfuse integration.
    """
    def __init__(self):

        # Setting up environment variables for Langfuse and OpenAI
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", ""),
        )
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "").strip()
        os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY", "").strip()


    @staticmethod
    def format_messages(system: str, user: str) -> list:
        """
        Formats messages for the LMClient chat method.
        Returns:
            :return list: Formatted messages.
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def chat_with_tools(self, messages: list, model_config: dict, gateway: GatewayToolCaller, max_steps: int = 8) -> str:
        """
        Chat completion con tool-calling:
        - passa i tool al modello
        - se il modello chiede tool_calls, li esegue (via gateway) e continua
        """
        for step in range(max_steps):
            response = litellm.completion(
                max_tokens=model_config["max_tokens"],
                messages=messages,
                model=model_config["name"],
                response_format=model_config.get("response_format"),
                temperature=model_config["temperature"],
                tools=model_config["tools"],
                tool_choice="auto",
                stream=False,
            )

            msg = response.choices[0].message

            # LiteLLM può restituire tool_calls in forme diverse a seconda del provider
            tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None) or []

            # Se non ci sono tool call, è la risposta finale
            if not tool_calls:
                final_text = msg.content or ""
                self.langfuse.create_event(
                    name="llm_final_answer",
                    input=messages,
                    output=final_text,
                    metadata={"model": model_config["name"], "steps": step},
                )
                self.langfuse.flush()
                return final_text

            # Aggiungi il messaggio assistant che contiene la richiesta tool
            # (formato OpenAI-style)
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id if hasattr(tc, "id") else tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc.function.name if hasattr(tc, "function") else tc["function"]["name"],
                            "arguments": tc.function.arguments if hasattr(tc, "function") else tc["function"]["arguments"],
                        }
                    }
                    for tc in tool_calls
                ],
            })

            # Esegui tool calls
            for tc in tool_calls:
                tc_id = tc.id if hasattr(tc, "id") else tc["id"]
                fn_name = tc.function.name if hasattr(tc, "function") else tc["function"]["name"]
                args_raw = tc.function.arguments if hasattr(tc, "function") else tc["function"].get("arguments", "{}")

                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}

                result = gateway.call(fn_name, args)

                # Inserisci la "tool observation" nella conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "content": json.dumps(result),
                })

        raise RuntimeError(f"Tool loop exceeded max_steps={max_steps}")
