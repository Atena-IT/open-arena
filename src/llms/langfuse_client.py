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
            host="http://localhost:3000",
        )
        os.environ["OPENAI_API_KEY"] = "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"
        print(os.environ["OPENAI_API_KEY"])


    def format_messages(self, system: str, user: str) -> list:
        """
        Formats messages for the LMClient chat method.
        Returns:
            :return list: Formatted messages.
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


    def chat(self, messages: list, model: str) -> str:
        """
        Sends a chat completion request to the selected model.
        Args:
            :param messages: List of message dicts.
            :param model:
        Returns:
            :return dict: The response from the model.

        """
        response = litellm.completion(model=model, messages=messages)

        # Tracing as Langfuse event
        event = self.langfuse.create_event(
            name="llm_completion",
            input=messages,
            output=response.choices[0].message.content,
            metadata={"model": model}
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
    user_prompt = "Ciao! Come stai?"
    print("SYSTEM: ", system_prompt)
    print("USER: ", user_prompt)
    messages = client.format_messages(system_prompt, user_prompt)
    assistant_response = client.chat(model="gpt-4.1-mini", messages=messages)
    print("ASSISTANT: ", assistant_response)
