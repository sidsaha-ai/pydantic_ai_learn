"""
This provides a function to fetch the LLM model to use.
"""

import requests
from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIModel


class LLMModel:
    model_type: str
    ollama_model_name: str

    _lm_studio_base_url: str = 'http://localhost:1234/v1'
    _lm_studio_api_key: str = 'lm_studio'

    def _fetch_model_name(self) -> str:
        url: str = f'{self._lm_studio_base_url}/models'

        r = requests.get(url, timeout=10)
        if not r.ok:
            raise Exception(f'Request to fetch model names failed: {r.content}')

        c = r.json()
        return c.get('data', []) and c.get('data', [])[0].get('id', None) or None

    def fetch_lm_studio_model(self) -> OpenAIModel:
        """
        Creates and returns a model for LM Studio.
        """
        client = AsyncOpenAI(
            base_url=self._lm_studio_base_url, api_key=self._lm_studio_api_key,
        )
        model = OpenAIModel(
            self._fetch_model_name(), openai_client=client,
        )
        return model
    
    def fetch_ollama_model(self) -> OllamaModel:
        """
        Creates and returns a model for Ollama.
        """
        model = OllamaModel(self.ollama_model_name)
        return model
    
    def fetch_model(self) -> Model:
        """
        Returns a model based on the parameters.
        """
        match self.model_type:
            case 'lm_studio':
                return self.fetch_lm_studio_model()
            case 'ollama':
                return self.fetch_ollama_model()
