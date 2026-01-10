from dotenv import load_dotenv
import litellm

from src.llms.llm_client import LLMClient

load_dotenv()

class LangfuseLLMClient(LLMClient):
    """
    LLM Client with Langfuse observability integration.
    
    Inherits all functionality from LLMClient and adds automatic
    tracing of LLM calls to Langfuse via LiteLLM callbacks.
    """
    
    def __init__(
        self
    ):
        """
        Initialize LangfuseLLMClient with Langfuse observability.
        """
        super().__init__()
        
        litellm.success_callback = ["langfuse_otel"]
        litellm.failure_callback = ["langfuse_otel"]

