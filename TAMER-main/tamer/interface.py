import os
import pygame
import speech_recognition as sr

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

        # Initialiser le moteur de reconnaissance vocale
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def get_coach_action(self):
        """
        Utilise la commande vocale pour recevoir un feedback.
        Retourne 1 pour 'yes' et 0 pour 'no'.
        """
        print("Veuillez donner un feedback vocal : 'yes' ou 'no'")
        with self.microphone as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=3)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Commande reconnue : {command}")
                if "yes" in command:
                    return 1  # Feedback positif
                    print("Hakim quel bg")
                elif "no" in command:
                    return 0  # Feedback négatif
                    print("Hakim quel bg")
            except sr.WaitTimeoutError:
                print("Aucune commande détectée dans le temps imparti.")
            except sr.UnknownValueError:
                print("Commande non reconnue. Essayez encore.")
            except sr.RequestError as e:
                print(f"Erreur avec le service de reconnaissance vocale : {e}")
        return None  # Aucun feedback
        
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
