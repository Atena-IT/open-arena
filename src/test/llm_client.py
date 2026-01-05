import json, litellm, os, time
from dotenv import load_dotenv
from langfuse import Langfuse


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


    async def chat_with_mcp_tools(self, messages: list, model_config: dict, mcp_session, mcp_tools_openai: list, max_steps: int = 8) -> str:
        """
        Run an **asynchronous** LiteLLM chat completion with tool-calling executed via **MCP**.
        Parameters:
            :param messages: List of OpenAI-style chat messages
            :param model_config: Model configuration
            :param mcp_session: An already-initialized MCP session used to invoke tools.
            :param mcp_tools_openai: Tools expressed in OpenAI schema
            :param max_steps: Maximum number of tool-calling iterations to prevent infinite loops.
        Return:
            return The model's final answer as a string.
        Raises:
            :exception RuntimeError: If the tool loop exceeds `max_steps`.
        """
        # Storing all tools outputs
        tool_calls_log = []  # list[{"tool": str, "content": str}]


        for step in range(max_steps):
            response = await litellm.acompletion(
                max_tokens=model_config["max_tokens"],
                messages=messages,
                model=model_config["name"],
                response_format=model_config.get("response_format"),
                stream=False,
                temperature=model_config["temperature"],
                tools=mcp_tools_openai,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Final step
            if not tool_calls:
                final_text = msg.content or ""
                # Tracing as Langfuse event and flushing
                self.langfuse.create_event(
                    name="llm_response",
                    input=messages,
                    output={"content": final_text, "tool_calls": tool_calls_log},
                    metadata={"model": model_config["name"], "steps": step},
                )
                self.langfuse.flush()
                return final_text

            # Tool calls done
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})
            for call in tool_calls:
                function_name = call.function.name
                args_raw = call.function.arguments or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}
                result = await mcp_session.call_tool(function_name, args)
                tool_calls_log.append({"tool": function_name, "content": str(result.content)})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": function_name, "content": str(result.content)})

        self.langfuse.flush()
        raise RuntimeError(f"Tool loop exceeded max_steps={max_steps}")
