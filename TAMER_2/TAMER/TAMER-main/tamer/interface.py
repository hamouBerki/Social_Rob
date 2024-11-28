import os
import pygame

class Interface:
    """ Pygame interface for training TAMER """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # Positionner la fenêtre pygame pour qu'elle ne chevauche pas l'environnement gym
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        # Création de la fenêtre
        self.screen = pygame.display.set_mode((200, 100))
        area = self.screen.fill((0, 0, 0))  # Remplir l'écran de noir
        pygame.display.update(area)

    def get_coach_action(self):
        """
        Cette méthode gère l'entrée du coach (utilisation des touches fléchées pour décider de l'action).
        """
        print("get_coach_action called")
        # Mettre à jour les événements
        pygame.event.pump()

        # Vérification des événements
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    print("Coach Action: LEFT")
                    return 0  # Action "gauche"
                elif event.key == pygame.K_RIGHT:
                    print("Coach Action: RIGHT")
                    return 1  # Action "droite"
        return None  # Si aucune action n'est donnée

    def show_action(self, action, is_coach=False):
        """
        Affiche l'action de l'agent ou du coach sur l'écran Pygame.
        Args:
            action: Action numérique.
            is_coach: Si True, cela met en évidence que l'action vient du coach.
        """
        color = (0, 255, 0) if is_coach else (255, 255, 255)  # Vert si c'est le coach, sinon blanc
        area = self.screen.fill((0, 0, 0))  # Effacer l'écran
        pygame.display.update(area)
        text = self.font.render(self.action_map[action], True, color)  # Texte de l'action
        text_rect = text.get_rect()
        text_rect.center = (100, 50)  # Placer le texte au centre
        area = self.screen.blit(text, text_rect)  # Afficher le texte
        pygame.display.update(area)  # Actualiser l'affichage
