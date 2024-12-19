"""
This would implement a language translator that would take an input text, detect the language, and
translate it to another language.
"""
import asyncio
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from utils.llm_model import LLMModel

from examples.translation.agents.language_detector import LanguageDetectorResult, agent as detector_agent
from examples.translation.agents.translator import TranslatorDependencies, TranslatorResult, agent as translator_agent

logfire.configure(console=False)

@dataclass
class LanguageTranslatorDeps:
    """
    The dependencies needed by this agent.
    """
    detector_agent: Agent
    translator_agent: Agent
    desired_language: str


class LanguageTranslatorResult(BaseModel):
    """
    The data model produced by this agent as the final result.
    """
    input_text: str = Field(
        description='The input text given to this agent.',
        examples=["The frog jumped out of the water and landed on the princess's lap."],
    )
    detected_language: str = Field(
        description='The detected language of the input text',
        examples=['English'],
    )
    desired_language: str = Field(
        description='The desired language to which the input text is translated to',
        examples=['Hindi'],
    )
    translated_text: str = Field(
        description='The final translated output text',
        examples=["मेंढक पानी से बाहर कूद गया और राजकुमारी की गोद में आ बैठा।"],
    )

# create the agent
m = LLMModel()
m.model_type = 'groq'
m.groq_model_name = 'llama3-groq-70b-8192-tool-use-preview'

llm_model = m.fetch_model()

agent = Agent(
    llm_model,
    deps_type=LanguageTranslatorDeps,
    result_type=LanguageTranslatorResult,
    system_prompt=(
        'You are language translator. '
        'You are given an input text and the desired language to translate to. '
        'You can use a tool to detect the language of the input text. '
        'You can use another tool to translate the input text to the desired language. '
        'Finally, you have to output the input text given to you, the language you detected of the input text, the desired language to translate to given to you, and the final output translated text. '
    ),
)

@agent.tool
async def detect_language(ctx: RunContext[LanguageTranslatorDeps], input_text: str) -> str:
    """
    Tool that takes the input text given to the agent to detect the language of the input text.
    """
    deps: LanguageTranslatorDeps = ctx.deps
    res = await deps.detector_agent.run(input_text)

    data: LanguageDetectorResult = res.data
    return data.detected_language


@agent.tool
async def translate_input_text(
    ctx: RunContext[LanguageTranslatorDeps],
    input_text: str,
    detected_language: str,
) -> str:
    """
    Tool that translates the input text from the detected language to the desired language.

    Parameters:
        input_text (str): The input text provided to this agent.
        detected_language (str): The detected language of the input text gathered from other tools.
    """
    deps: LanguageTranslatorDeps = ctx.deps

    translator_deps = TranslatorDependencies(
        input_language=detected_language, desired_language=deps.desired_language,
    )
    res = await deps.translator_agent.run(input_text, deps=translator_deps)

    data: TranslatorResult = res.data
    return data.translated_text


async def main(input_text: str, desired_language: str) -> None:
    """
    The main function to execute.
    """
    deps = LanguageTranslatorDeps(
        detector_agent=detector_agent, translator_agent=translator_agent, desired_language=desired_language,
    )
    res = await agent.run(input_text, deps=deps)

    print(res.data)


if __name__ == '__main__':
    text: str = "The frog jumped out of the water and landed on the princess's lap."
    desired_lang: str = 'French'

    asyncio.run(
        main(text, desired_lang),
    )

