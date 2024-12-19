"""
An agent to detect the language of the input text.
"""
import asyncio

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from utils.llm_model import LLMModel

logfire.configure()


class LanguageDetectorResult(BaseModel):
    """
    The data model produced as the final result of this agent.
    """
    detected_language: str = Field(
        description='The detected language in English of the input text given to the agent for detection.',
        examples=['English', 'Hindi', 'Arabic'],
    )


# create the agent
m = LLMModel()
m.model_type = 'groq'
m.groq_model_name = 'llama3-groq-70b-8192-tool-use-preview'

llm_model = m.fetch_model()

agent = Agent(
    llm_model,
    result_type=LanguageDetectorResult,
    system_prompt=(
        'You are a language detector.'
        'Your only job is to detect the language of the text given to you and return the name of the language in English'
        'You should ONLY output the language you detect and nothing else.'
        'Example 1:'
        'Input Text: I am doing well today. How are you?'
        'Output: English'
        'Example 2:'
        'Input Text: आज मैं अच्छा हूँ। आप कैसे हैं?'
        'Output: Hindi'
    )
)


async def main(input_text: str) -> None:
    """
    The main function to execute to test this agent.
    """
    result = await agent.run(input_text)
    print(f'{input_text} : {result.data}')


if __name__ == '__main__':
    input_texts = [
        "The frog jumped out of the water and landed on the princess's lap.",
        "मेंढक पानी से बाहर कूद गया और राजकुमारी की गोद में आ बैठा।",
        "قفز الضفدع خارج الماء وحط في حضن الأميرة.",
        "ব্যাঙটি পানির বাইরে লাফিয়ে পড়ল এবং রাজকুমারীর কোলে গিয়ে পড়ল।",
        "Der Frosch sprang aus dem Wasser und landete auf dem Schoß der Prinzessin.",
    ]
    for t in input_texts:
        asyncio.run(
            main(t)
        )
