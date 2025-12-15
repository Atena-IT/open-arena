import litellm, os
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


    def chat(self, messages: list, model_config: dict) -> str:
        """
        Sends a chat completion request to the selected model.
        Args:
            :param messages: List of message dicts.
            :param model_config: Configuration dictionary for the model.
        Returns:
            :return dict: The response from the model.
        """
        response = litellm.completion(
            max_tokens=model_config["max_tokens"],
            messages=messages,
            model=model_config["name"],
            response_format=model_config["response_format"],
            temperature=model_config["temperature"],
            tools=model_config["tools"],
            stream=model_config["stream"],
        )

        # Tracing as Langfuse event
        self.langfuse.create_event(
            name="llm_completion",
            input=messages,
            output=response.choices[0].message.content,
            metadata={"model": model_config["name"]}
        )

        # Optional flush to be sure to send the message immediately
        self.langfuse.flush()

        return response.choices[0].message.content


""" MAIN """
if __name__ == "__main__":

    # LLM Client
    client = LLMClient()

    # Conversation
    system_prompt = "You are a helpful assistant."
    user_prompt = "Ciao! Mi spieghi la teoria della relatività?"
    print("SYSTEM: ", system_prompt)
    print("USER: ", user_prompt)
    messages = client.format_messages(system_prompt, user_prompt)
    assistant_response = client.chat(messages=messages,
                                     model_config= {
                                         "name": "huggingface/Qwen/Qwen3-Next-80B-A3B-Instruct",
                                         "max_tokens": 2048,
                                         "response_format": None,
                                         "temperature": 0.3,
                                         "tools": [],
                                         "stream": False,
                                     })
    print("ASSISTANT: ", assistant_response)
