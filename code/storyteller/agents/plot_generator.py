"""
This is the Plot Generator agent that generates the high level plot of the story.
"""
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from utils.llm_model import LLMModel

logfire.configure(console=False, scrubbing=False)


@dataclass
class PlotDeps:
    """
    The dependencies needed by this agent.
    """
    theme: str  # the high level theme of the story


class PlotResult(BaseModel):
    """
    The data model produced as the final result of this agent.
    """
    characters: list[str] = Field(description='The characters of the story.')
    setting: str = Field(description="The story's general setting")
    inciting_incident: str = Field(description='The inciting incident in the plot.')
    action: str = Field(description='The action to the incident.')
    climax: str = Field(description='The climax of the plot.')
    resolution: str = Field(description='The resolution of the plot of the story.')


m = LLMModel()
m.model_type = 'groq'
m.groq_model_name = 'llama3-groq-70b-8192-tool-use-preview'

agent = Agent(
    m.fetch_model(),
    deps_type=PlotDeps,
    result_type=PlotResult,
    system_prompt=(
        'You are on the plot generater of a story based on a given theme. '
        'Your job is to generate a high-level plot of the story. '

        'You should output the characters of the story. The number of characters in the story should be between 3 and 7. '
        'Do not use generic names for the characters. Bring in more diversity in the type of characters.'

        'You should output the general setting where the story takes place, an inciting incident that would happen in the story, '
        'an action to the incident, a climax, and a resolution to the story. '
        'Make sure that the general setting is a bit elaborate and has depth and is not too generic. Be creative.'
        'The action to the incident and the climax could use twists to the plot to make it more entertaining'
        'Also, the resolution to the story should be complex and not one-dimensional.'

        'Remember, your job is to only generate a high-level plot of the story.'
        'Ensure that all the characters have a role in the story. '
        'The goal is to create a high-level plot that is engaging, realistic, and entertaining. '
    ),
)


@agent.system_prompt
def theme_prompt(ctx: RunContext[PlotDeps]) -> str:
    """
    Add the theme as a system prompt.
    """
    deps: PlotDeps = ctx.deps
    return f'The theme of the story is a {deps.theme}'


if __name__ == '__main__':
    # theme: str = 'murder mystery'
    theme: str = 'romantic love story'
    # theme: str = 'thriller'
    # theme: str = 'drama'

    res = agent.run_sync("Let's start!", deps=PlotDeps(theme=theme))
    print(res.data)
