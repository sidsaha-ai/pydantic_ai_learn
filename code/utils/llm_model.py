"""
This provides a function to fetch the LLM model to use.
"""

import requests
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel


def _base_url() -> str:
    """
    Returns the base URL of the LLM.
    """
    return 'http://localhost:1234/v1'

def _api_key() -> str:
    """
    Returns the API Key.
    """
    return 'lm_studio'


def _fetch_model_name() -> str:
    url: str = f'{_base_url()}/models'
    
    r = requests.get(url)
    if not r.ok:
        raise Exception(f'Request to fetch model names failed: {r.content}')
    
    c = r.json()
    return c.get('data', []) and c.get('data', [])[0].get('id', None) or None


def fetch_model() -> OpenAIModel:
    """
    This returns an OpenAIModel.
    """
    client = AsyncOpenAI(base_url=_base_url(), api_key=_api_key())
    model = OpenAIModel(
        _fetch_model_name(), openai_client=client,
    )
    return model
