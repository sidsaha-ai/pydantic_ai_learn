"""
This agent generates one scene of the short story.
"""
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from storyteller.agents.character_richness import CharacterRichnessResult
from storyteller.agents.plot_generator import PlotResult
from utils.llm_model import LLMModel

logfire.configure(console=False, scrubbing=False)

@dataclass
class SceneDeps:
    plot: PlotResult
    characters: CharacterRichnessResult
    # TODO: add previous scene and overall summary of the story so far.


class SceneResult(BaseModel):
    scene: str

    def __str__(self) -> str:
        return f'**Scene**: {self.scene}'


m = LLMModel()
m.model_type = 'groq'
m.groq_model_name = 'llama3-groq-70b-8192-tool-use-preview'

agent = Agent(
    m.fetch_model(),
    deps_type=SceneDeps,
    result_type=SceneResult,
    system_prompt=(
        'Your job is to write ONE scene of the short story that we are writing. '
        'You are given the overall high-level plot information, the characters of the story and the details about their personalities, the summary of the story so far, and the immediate previous scene. '
        'Using the information you have been given, you have to generate the next logical scene of the overall story. '
        'A scene is a self-contained segment of the story that unfolds in a specific location, involves one or more characters, and progresses the narrative by revealing key events, interactions, or conflicts. '
        '\n'
        'Remember to produce the scene in the form of literature, as it will be part of the overall story. '
        'Be creative. Do not be monotonous. Use the characters creatively, and add dialogues between them as you see fit. '
        'Remember that not all characters have to be used in a scene. '
        'If there is no summary of the story so far, that means that this is the first scene of the story. Move smoothly (and not abruptly between scenes.). '
        '\n'
    ),
)

@agent.system_prompt
def plot(ctx: RunContext[SceneDeps]) -> str:
    """
    Adds the plot to the system prompt.
    """
    deps: SceneDeps = ctx.deps
    prompt: str = (
        'Below is the high-level plot of the story: \n'
        f'{deps.plot}'
        '\n'
    )
    return prompt

@agent.system_prompt
def characters(ctx: RunContext[SceneDeps]) -> str:
    """
    Adds the character's traits to the prompt.
    """
    deps: SceneDeps = ctx.deps
    prompt: str = (
        'Below are the characters of the story and their details: \n'
        f'{deps.characters}'
    )
    return prompt
