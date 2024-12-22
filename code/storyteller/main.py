"""
This is the main function that uses all the agents.
"""
from storyteller.agents.character_richness import (CharacterDeps,
                                                   CharacterRichnessResult)
from storyteller.agents.character_richness import \
    agent as character_richness_agent
from storyteller.agents.plot_generator import PlotDeps, PlotResult
from storyteller.agents.plot_generator import agent as plot_generator_agent
from storyteller.agents.scene_generator import SceneDeps, SceneResult
from storyteller.agents.scene_generator import agent as scene_generator_agent


def main():
    """
    The main function that uses all the agents to generate a full-story.
    """
    theme: str = 'a murder mystery'  # the theme of the story

    # generate the plot
    res = plot_generator_agent.run_sync("Let's start!", deps=PlotDeps(theme=theme))
    plot: PlotResult = res.data
    print('=== Plot ===')
    print(plot)
    print()

    # generate character richness
    res = character_richness_agent.run_sync(
        'Generate character richness for the story.',
        deps=CharacterDeps(theme=theme, plot=plot),
    )
    characters: CharacterRichnessResult = res.data
    print('=== Character Richness ===')
    print(characters)
    print()

    # generate a scene
    res = scene_generator_agent.run_sync(
        'Generate the next scene of the story.',
        deps=SceneDeps(plot=plot, characters=characters),
    )
    print('=== Scene ===')
    scene: SceneResult = res.data
    print(scene)


if __name__ == '__main__':
    main()
