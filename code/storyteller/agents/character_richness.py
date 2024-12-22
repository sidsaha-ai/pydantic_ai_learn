"""
This is the character richness agent that given a brief about a character, generates
more detailed info about the character.
"""
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from utils.llm_model import LLMModel

from storyteller.agents.plot_generator import PlotResult

logfire.configure(console=False, scrubbing=False)

@dataclass
class CharacterDeps:
    """
    The dependencies for this agent.
    """
    theme: str  # the high level theme of the story
    plot: PlotResult  # the overall plot of the story


class CharacterTraits(BaseModel):
    """
    This model contains the character traits of a particular character.
    """
    name: str = Field(description='The name of the character.')
    background: str = Field(description='The brief history of the character generated by this agent.')
    personality: str = Field(description='The core personality traits of the character generated by this agent.')
    motivations: str = Field(description='The motivations of this character in the story generated by this agent.')
    internal_conflicts: str = Field(description='The internal conflicts of this character generated by this agent.')
    external_conflicts: str = Field(description='The external conflicts of this character generated by this agent.')
    relationships: str = Field(description='The relationship of this character with other characters in the story generated by this agent.')


class CharacterRichnessResult(BaseModel):
    """
    The final character richness result generated by this agent.
    """
    character_traits: list[CharacterTraits] = Field(description='A list of traits of each character in the story.')


m = LLMModel()
m.model_type = 'groq'
m.groq_model_name = 'llama3-groq-70b-8192-tool-use-preview'

system_prompt: str = (
    'You are a character development agent responsible for adding depth and richness to the '
    'characters in a story. You will receive the list of characters and the overall plot of the '
    'story, and your task is to generate detailed background information and personality traits '
    'of each character. These background and personality traits will be used in the detailed story.\n\n'
    'For each character, you should generate the following: \n'
    '1. **Name**: The name of the character. \n'
    '2: **Background**: A brief history of the character, including their upbringing, family, education, and significant events in their life that shaped them.\n'
    "3. **Personality**: Describe the character's core personality traits (e.g., introverted, optimistic, selfish, compassionate).\n"
    '4. **Motivations**: What does this character want most in the story? What are they striving for? What are their fears or desires?\n'
    '5. **Internal Conflict**: Does this character have any inner struggles or dilemmas that might influence their actions?\n'
    '6. **External Conflict**: How do external forces (such as other characters, the environment, or situations) challenge this character and influence their decisions?\n'
    "7. **Relationships**: What is the character's relationship with the other characters in the story? Are they allies, rivals, or indifferent to each other?\n\n"
    'Remember, the goal is to add depth to each character so that they are more than just plot devices—they should feel like real, complex individuals whose actions make sense based on their backgrounds and personalities.\n\n'
    'Use the information from the plot generator (which you will receive as context) to tailor your character development to fit the genre and tone of the story. Ensure that each character has a unique and interesting background and personality that adds value to the overall plot.\n\n'
)

agent = Agent(
    m.fetch_model(),
    deps_type=CharacterDeps,
    result_type=CharacterRichnessResult,
    system_prompt=system_prompt,
)

@agent.system_prompt
def plot_prompt(ctx: RunContext[CharacterDeps]) -> str:
    """
    Add context-driven system prompt.
    """
    deps: CharacterDeps = ctx.deps
    prompt = (
        f'Here is some more context about the plot. The theme of the story is {deps.theme}.\n\n'
        'Below is more information about the characters and the plot.\n'
        f"1. **Characters**: {', '.join(deps.plot.characters)}\n"
        f'2. **Setting of the story**: {deps.plot.setting}\n'
        f'3. **Incident Incident in the story**: {deps.plot.inciting_incident}\n'
        f'4: **Action to the incident**: {deps.plot.action}\n'
        f'5: **Climax of the story**: {deps.plot.climax}\n'
        f'6: **Resolution of the climax**: {deps.plot.resolution}\n\n'
        'Keep the plot in mind when working on your task to generate character traits.'
    )
    return prompt