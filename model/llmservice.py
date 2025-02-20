from zhipuai import ZhipuAI
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import os

def client(service="doubao", api_key=None):
    """Initialize LLM client based on service type.
    
    Args:
        service: Service type ('doubao', 'openai', or 'deepseek')
        api_key: Optional API key override
        
    Returns:
        LLM client instance
    """
    if service == "doubao":
        ark_key = api_key or os.environ.get("ARK_API_KEY")
        if not ark_key:
            raise ValueError("ARK_API_KEY environment variable not set")
        return Ark(api_key=ark_key)
    
    elif service == "openai":
        if api_key:
            return OpenAI(api_key=api_key)
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=openai_key)
    
    elif service == "deepseek":
        deepseek_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com"
        )
    
    else:
        raise ValueError(f"Unsupported service: {service}")
