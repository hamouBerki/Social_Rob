from pathlib import Path
import uuid
import cv2
import time
from itertools import count
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define LOGS_DIR globally
LOGS_DIR = Path(__file__).parent.joinpath('logs')
LOGS_DIR.mkdir(exist_ok=True)  # Create the directory if it doesn't exist


class SGDFunctionApproximator:
    """SGD function approximator with RBF preprocessing."""

    def __init__(self, env):
        # Feature preprocessing: Normalize to zero mean and unit variance
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype="float64"
        )
        self.scaler = StandardScaler()
        self.scaler.fit(observation_examples)

        # RBF Kernel features
        self.featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # Models for each action
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if action is None:
            return [model.predict([features])[0] for model in self.models]
        return self.models[action].predict([features])[0]

    def update(self, state, action, target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [target])

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    """
    TAMER Agent for Human-in-the-loop Reinforcement Learning.
    """

    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0,
        min_eps=0,
        tame=True,
        ts_len=0.2,
        alpha=0.5,
        output_dir=LOGS_DIR,
        model_file_to_load=None,
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.tame = tame
        self.ts_len = ts_len
        self.alpha = alpha
        self.output_dir = output_dir

        # Model initialization
        if model_file_to_load:
            self.load_model(model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env)

        # Epsilon decay
        self.epsilon_step = (epsilon - min_eps) / num_episodes

    def act(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if np.random.random() > self.epsilon:
            return np.argmax(self.H.predict(state))
        return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp):
        """Train the agent for a single episode."""
        print(f"Episode {episode_index + 1}:")
        state, _ = self.env.reset()
        total_reward = 0

        for t in count():
            action = self.act(state)
            if self.tame:
                disp.show_action(action)

            next_state, env_reward, done, _, _ = self.env.step(action)

            # Human feedback
            human_feedback = 0
            if self.tame:
                now = time.time()
                while time.time() < now + self.ts_len:
                    human_feedback = disp.get_scalar_feedback()
                    if human_feedback != 0:
                        break

            # Combine feedback and environment reward
            combined_reward = self.alpha * human_feedback + (1 - self.alpha) * env_reward
            self.H.update(state, action, combined_reward)

            total_reward += env_reward
            state = next_state

            if done:
                print(f"  Total reward: {total_reward}")
                break

        # Epsilon decay
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save=None):
        """Main training loop."""
        from .interface import Interface  # Import the human feedback interface

        disp = Interface(action_map={0: "left", 1: "none", 2: "right"})

        for i in range(self.num_episodes):
            self._train_episode(i, disp)

        self.env.close()
        if model_file_to_save:
            self.save_model(model_file_to_save)

    def save_model(self, filename):
        """Save the H function model."""
        model_file = self.output_dir / filename
        with open(model_file, "wb") as f:
            pickle.dump(self.H, f)
        print(f"Model saved to {model_file}")

    def load_model(self, filename):
        """Load a pre-trained H function model."""
        model_file = self.output_dir / filename
        with open(model_file, "rb") as f:
            self.H = pickle.load(f)
        print(f"Model loaded from {model_file}")
