"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio  #programmation asynchrone
# import gym
import gymnasium as gym

from tamer.agent import Tamer


async def main():
    # env = gym.make('MountainCar-v0', render_mode='human')
    env = gym.make('MountainCar-v0', render_mode='rgb_array')           # creation de l'environement, render c'est pour les frames




    # env = gym.make('MountainCar-v0', 
    #                render_mode="human", 
    #                apply_api_compatibility=True)
    # print(env.reset())
    # hyperparameters
    discount_factor = 1   #gamma = reponses futur %R reponses immédiates
    
    epsilon = 0  # vanilla Q learning actually worksw well with no random exploration
                 #Tamer utilise le  feedback humain donc l'exploration aléatoire n'est pas nécessaire ==> epsilon = 0
    
    min_eps = 0
    num_episodes = 2     # l'agent s'entraine 2 episodes 
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.3 #0.3 sen entre chaque acrion de l'agent

    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)  #creation de l'agent

    print("Awaiting agent.train")

    await agent.train(model_file_to_save='autosave') #on appelle la methode train

    print("Starting agent.play")
    agent.play(n_episodes=1, render=True)
    agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())




