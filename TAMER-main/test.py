import gymnasium as gym

# Création de l'environnement LunarLander-v3
env = gym.make('LunarLander-v3', render_mode="human")

# Réinitialisation de l'environnement pour obtenir la première observation
observation, info = env.reset()

# Boucle pour 1000 étapes
for _ in range(1000):
    # Choisir une action aléatoire
    action = env.action_space.sample()
    
    # Exécuter l'action dans l'environnement
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Vérifier si l'épisode est terminé ou tronqué
    if terminated or truncated:
        # Réinitialiser l'environnement
        observation, info = env.reset()

# Fermer l'environnement après la simulation
env.close()
