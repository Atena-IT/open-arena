import json, litellm, os
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
        Workflow:
        1) Sends `messages` to the model using `litellm.acompletion`, passing MCP tools already converted to **OpenAI
            tool format** (`mcp_tools_openai`).
        2) If the model returns `tool_calls`, appends the assistant message to the conversation, executes each requested
            tool via `await mcp_session.call_tool(tool_name, args)`, and appends a corresponding `role="tool"` message
            containing the tool output.
        3) Repeats until the model produces a final response with no tool calls, or until `max_steps` is reached.
        Parameters:
            :param messages: List of OpenAI-style chat messages
            :param model_config: Model configuration
            :param mcp_session: An already-initialized MCP session used to invoke tools.
            :param mcp_tools_openai: Tools expressed in OpenAI schema
            :param max_steps: Maximum number of tool-calling iterations to prevent infinite loops.
        Returns:
            :return The model's final answer as a string.
        Raises:
            :exception RuntimeError: If the tool loop exceeds `max_steps`.
        """
        for step in range(max_steps):
            response = await litellm.acompletion(
                max_tokens=model_config["max_tokens"],
                messages=messages,
                model=model_config["name"],
                response_format=model_config.get("response_format"),
                temperature=model_config["temperature"],
                tools=mcp_tools_openai,
                tool_choice="auto",
                stream=False,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
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
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                fn_name = call.function.name
                args_raw = call.function.arguments or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}
                result = await mcp_session.call_tool(fn_name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "content": str(result.content),
                    }
                )

        raise RuntimeError(f"Tool loop exceeded max_steps={max_steps}")
