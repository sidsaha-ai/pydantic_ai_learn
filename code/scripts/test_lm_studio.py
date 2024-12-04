from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from utils import llm_model

def main():
    model = llm_model.fetch_model()
    agent = Agent(
        model,
        system_prompt='Be concise, reply with only one sentence.',
    )
    
    result = agent.run_sync('Where does "Hello, World!" come from?')
    print(result.data)


if __name__ == '__main__':
    main()
