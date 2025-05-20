import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLMs
def get_anthropic_llm(model="claude-3-7-sonnet-20250219"):
    """Get an Anthropic LLM instance."""
    return ChatAnthropic(model=model)


def get_openai_llm(model="gpt-4o-mini", temperature=0, json_mode=False):
    """Get an OpenAI LLM instance with optional JSON mode."""
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        model_kwargs=model_kwargs
    ) 
