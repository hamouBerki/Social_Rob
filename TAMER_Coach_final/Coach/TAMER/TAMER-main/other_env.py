import asyncio
import gymnasium as gym
from tamer.agent import Tamer
import cv2
import pygame
import matplotlib.pyplot as plt

CARTPOLE_ACTION_MAP = {0: 'left', 1: 'right'}

async def main():
    # Création de l'environnement CartPole
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Hyperparamètres
    discount_factor = 1
    epsilon = 0
    min_eps = 0
    num_episodes = 3
    tame = True
    tamer_training_timestep = 0.2

    # Création de l'interface pygame pour feedback humain
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption('TAMER Feedback')
    font = pygame.font.Font(None, 36)

    # Création de l'agent TAMER pour CartPole
    agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load=None)

    # Vérification du type de l'agent pour s'assurer qu'il s'agit bien de Tamer
    print(f"Agent type: {type(agent)}")  # Devrait afficher: <class 'tamer.agent.Tamer'>

    # Vérification de la présence de la méthode 'train' dans l'agent
    if hasattr(agent, 'train'):
        print("La méthode 'train' est disponible.")
    else:
        print("Erreur: 'train' non trouvé dans l'agent.")

    # Entraînement de l'agent avec feedback humain
    print("Awaiting agent.train")
    await agent.train(model_file_to_save='autosave')

    # Phase de jeu pour observer le comportement de l'agent
    print("Starting agent.play")
    agent.play(n_episodes=1, render=True)

    # Évaluation des performances de l'agent sur plusieurs épisodes
    ep_rewards=agent.evaluate(n_episodes=300)
    

    plt.figure(figsize=(10, 6))
    plt.plot(ep_rewards, label='Récompenses par épisode', color='blue')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    plt.title('Évolution des récompenses au fil des épisodes')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Quitter pygame
    pygame.quit()


if __name__ == '__main__':
    asyncio.run(main())
