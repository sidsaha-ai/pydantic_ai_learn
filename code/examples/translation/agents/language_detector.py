"""
This agent would detect the language of the input string.
"""
import argparse
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from utils import llm_model
import asyncio


@dataclass
class Dependencies:
    text: str

class Outputs(BaseModel):
    language: str = Field(description='The detected language of the input text.')


class AgentMaker:

    @staticmethod
    def _fetch_system_prompt() -> str:
        system_prompt: str = """
        You are a language detector. You need to detect the language of the text provided and return the detected language (in English). \
        Respond with ONLY the detected language (single word response), and nothing else.
        """
        return system_prompt.strip().strip('\n')

    @staticmethod
    def make_agent() -> Agent:
        model = llm_model.fetch_model()
        agent = Agent(
            model,
            deps_type=Dependencies,
            result_type=Outputs,
            system_prompt=AgentMaker._fetch_system_prompt(),
            retries=3,
        )
        return agent

async def main(input_text):
    deps: Dependencies = Dependencies(text=input_text)
    agent: Agent = AgentMaker.make_agent()

    result = await agent.run(input_text)
    print(result.data)


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
    