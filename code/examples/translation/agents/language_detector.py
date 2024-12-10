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


class AgentMaker:
    """
    Class to make and return a language detector agent.
    """

    @staticmethod
    def _fetch_system_prompt() -> str:
        system_prompt: str = """
        You are a language detector. You need to detect the language of the text provided and return the detected language (in English). \
        Respond with ONLY the detected language.
        """
        return system_prompt.strip().strip('\n')

    @staticmethod
    def make_agent() -> Agent:
        """
        Function to create and return the language detector agent.
        """
        m = LLMModel()
        m.model_type = 'ollama'
        m.ollama_model_name = 'llama3.1:8b'

        model = m.fetch_model()
        agent = Agent(
            model,
            result_type=Outputs,
            system_prompt=AgentMaker._fetch_system_prompt(),
            retries=3,
        )
        return agent


async def main(input_text):
    """
    The main function to test this agent.
    """
    agent: Agent = AgentMaker.make_agent()

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
