"""
This agent would detect the language of the input string.
"""
import argparse
import asyncio
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from utils.llm_model import LLMModel

logfire.configure()


class Outputs(BaseModel):
    """
    The outputs of the language detector agent.
    """
    language: Literal[
        'English',
        'Hindi',
        'Arabic',
        'Odia',
        'French',
        'German',
        'Bengali',
    ] = Field(description='The detected language of the input text.')


m = LLMModel()
m.model_type = 'ollama'
m.ollama_model_name = 'llama3.1:8b'

model = m.fetch_model()

agent = Agent(
    model,
    result_type=Outputs,
    system_prompt='You are a language detector. You need to detect the language of the \
        text provided and returnthe detected language (in English). Respond wth ONLY the detected language.',
    retries=3,
)


async def main(input_text):
    """
    The main function to test this agent.
    """
    result = await agent.run(input_text)
    print(f'Language Detected: {result.data.language}')


# test the agent
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text', type=str, required=True,
    )
    args = parser.parse_args()

    asyncio.run(
        main(args.input_text)
    )
