"""
This agent generates one scene of the short story.
"""
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from storyteller.agents.character_richness import CharacterRichnessResult
from storyteller.agents.plot_generator import PlotResult
from utils.llm_model import LLMModel


@dataclass
class SceneDeps:
    plot: PlotResult
    characters: CharacterRichnessResult
    scene_num: int
    # TODO: add previous scene and overall summary of the story so far.


m = LLMModel()
m.model_type = 'groq'
m.groq_model_name = 'llama3-groq-70b-8192-tool-use-preview'

agent = Agent(
    m.fetch_model(),
    deps_type=SceneDeps,
    result_type=str,
    system_prompt=(
        "Your job is to write ONE scene for the short story that we are collaboratively creating. A scene is a **self-contained narrative segment** that unfolds in a specific location, involves one or more characters, and moves the story forward by revealing key events, interactions, or conflicts. Your goal is to create a scene that feels vivid, immersive, and engaging, while fitting seamlessly into the overall plot.\n"
        '\n'
        "To achieve this:\n"
        "1. **Set the Scene**: Start with a detailed description of the location and atmosphere, incorporating sensory details (e.g., sights, sounds, smells, and emotions) to draw the reader in. Ensure the setting reflects the story's mood and genre.\n"
        "2. **Character Depth**: Use characters creatively, allowing their personalities, emotions, and internal conflicts to shine through their actions, dialogue, and body language. Avoid shallow interactions—focus on subtle, layered exchanges that reveal character dynamics.\n"
        "3. **Progress the Narrative**: Each scene should introduce or develop key plot points, conflicts, or mysteries. Avoid summarizing events—show them unfolding naturally through dialogue, actions, and discoveries.\n"
        "4. **Tension and Subtext**: Create tension by hinting at underlying secrets, conflicts, or motivations. Use subtext in dialogue and interactions to make the scene more intriguing.\n"
        "5. **Immersive Writing**: Write as if this is a final draft of a published work. Use rich, literary language, avoiding clichés and monotony. Maintain a smooth flow that transitions seamlessly between actions, dialogues, and descriptions.\n"
        "6. **Creativity and Balance**: Feel free to experiment with character dynamics or unexpected twists, but ensure the scene remains believable and aligns with the story's tone and trajectory. Not all characters need to appear in every scene.\n"
        '\n'
        "Additional Guidelines:\n"
        "- Use dialogue where appropriate, ensuring it feels natural and true to each character’s voice.\n"
        "- Avoid abrupt transitions between scenes—each should flow naturally from the last, even if unresolved threads remain for later exploration.\n"
        "- Be concise but detailed—aim for a scene length that feels substantial yet focused, avoiding unnecessary filler.\n"
        '\n'
        "You will receive:\n"
        "1. **The high-level plot**: A summary of the overall story and its key elements.\n"
        "2. **The characters**: A list of characters, along with their personalities, motivations, and conflicts.\n"
        "3. **The summary of the story so far**: What has happened up until this point.\n"
        "4. **The immediate previous scene**: The last scene to ensure continuity.\n"
        '\n'
        "Using this information, write the **next logical scene** in the form of literature, as it will be part of the complete short story. Be imaginative, precise, and compelling.\n"
        "It is extremely important to write the scene in a literay fashion so that it can be used as part of a larger story. \n"
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

@agent.system_prompt
def scene_num(ctx: RunContext[SceneDeps]) -> str:
    """
    Adds the scene number to the prompt.
    """
    deps: SceneDeps = ctx.deps
    prompt: str = (
        f'This is the scene number {deps.scene_num} of the overall story.\n'
        'Remember that the scene should read like a part of a story and should have literary finnesse and depth. The story is British, so write it in a British contemporary style.\n'
        'The scene should not be overly long, but not very short either. Each scene could be about 250 words.\n'
    )
    return prompt