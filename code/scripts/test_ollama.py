"""
Sample file to try out using Pydantic Agent with Ollama.
"""
from pydantic_ai import Agent
from utils.llm_model import LLMModel


def main():
    """
    The main function to start execution.
    """
    m = LLMModel()
    m.model_type = 'ollama'
    m.ollama_model_name = 'llama3.1:8b'

    model = m.fetch_model()
    agent = Agent(
        model,
        system_prompt='Be concise, reply with only one sentence.',
    )

    result = agent.run_sync('Where does "Hello, World!" come from?')
    print(result.data)


if __name__ == '__main__':
    main()
