import asyncio
import gymnasium as gym
from tamer.agent import Tamer
import cv2
import pygame

# Define a new action mapping for LunarLander if needed (it has 4 discrete actions)
LUNARLANDER_ACTION_MAP = {0: 'do nothing', 1: 'fire left engine', 2: 'fire main engine', 3: 'fire right engine'}

async def main():
    # Creation of the LunarLander-v3 environment
    env = gym.make('LunarLander-v3', render_mode='rgb_array')

    # Hyperparameters (adjusted for LunarLander)
    discount_factor = 0.99  # Since this is a more complex environment
    epsilon = 0  # Allows for some exploratory moves in early stages
    min_eps = 0
    num_episodes = 5  # More episodes due to increased complexity
    tame = True
    tamer_training_timestep = 0.5  # Can be longer to give the human trainer more time to react

    # Create the pygame interface for human feedback
    pygame.init()
    screen = pygame.display.set_mode((600, 200))
    pygame.display.set_caption('TAMER Feedback for LunarLander')
    font = pygame.font.Font(None, 36)

    # Create the TAMER agent for LunarLander
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)

    # Training the agent with human feedback
    print("Awaiting agent.train")
    await agent.train(model_file_to_save='autosave_lunarlander')

    # Phase to observe the agent's behavior
    print("Starting agent.play")
    agent.play(n_episodes=5, render=True)

    # Evaluate the agent's performance over multiple episodes
    agent.evaluate(n_episodes=30)

    # Quit pygame
    pygame.quit()

if __name__ == '__main__':
    asyncio.run(main())
