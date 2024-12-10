"""
This agent would translate the input string in a source language to another language.
"""
import argparse
import asyncio
from dataclasses import dataclass

import logfire
from examples.translation.agents import language_detector
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from utils.llm_model import LLMModel

logfire.configure()


@dataclass
class Deps:
    """
    The dependencies needed by this agent.
    """
    lang_detector_agent: Agent
    target_lang: str
    source_text: str


class Outputs(BaseModel):
    """
    The outputs produced by this agent.
    """
    translated_text: str = Field(description='The translated text result produced by the agent.')


m = LLMModel()
m.model_type = 'ollama'
m.ollama_model_name = 'llama3.1:8b'

model = m.fetch_model()

agent = Agent(
    model,
    result_type=Outputs,
    deps_type=Deps,
    system_prompt='You are a language translator. You job is to translate the given source text from \
        the source language to a target text in the target language. You should return the source text \
            (which needs to be translated), the source language, the target language, and the translated text.',
    retries=3,
)


@agent.system_prompt
async def source_language(ctx: RunContext[Deps]) -> str:
    """
    Returns a dynamic system prompt for the agent based on the
    result of the language detector.
    """
    r = await ctx.deps.lang_detector_agent.run(ctx.deps.source_text)
    return f'You have to convert the text from {r.data.language} to {ctx.deps.target_lang}.'


async def main(input_text: str, target_lang: str) -> None:
    """
    The main function to execute.
    """
    deps = Deps(
        lang_detector_agent=language_detector.agent,
        target_lang=target_lang,
        source_text=input_text,
    )

    result = await agent.run(input_text, deps=deps)

    print(result.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_text', required=True, type=str,
    )
    parser.add_argument(
        '--target_lang', required=True, type=str,
    )
    args = parser.parse_args()

    asyncio.run(
        main(args.input_text, args.target_lang),
    )
