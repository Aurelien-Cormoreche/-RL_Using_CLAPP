import torch
import torch.nn as nn
import random
from .models import ActorModel, CriticModel, Discrete_Maze_Model

class AC_Agent(nn.Module):
    """
    Actor-Critic Agent for reinforcement learning.
    This agent uses both an actor (policy) and a critic (value function) to learn and make decisions.
    The actor selects actions, while the critic evaluates the quality of those actions.
    """

    def __init__(
        self,
        num_features,
        num_action,
        activation,
        encoder,
        normalize_features=False,
        have_critic=True,
        two_layers=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # --- Parameters ---
        # num_features: number of input features (state representation size)
        # num_action: number of possible actions
        # activation: activation function for the critic network
        # encoder: neural network that encodes raw observations into features
        # normalize_features: whether to normalize the features using LayerNorm
        # have_critic: whether to include a critic network
        # two_layers: whether to use two layers in the actor and critic networks

        self.normalize_features = normalize_features
        self.encoder = encoder  # Encodes raw observations into feature vectors

        # --- Initialize actor and critic networks ---
        self.actor = ActorModel(num_features, num_action, two_layers)
        if have_critic:
            self.critic = CriticModel(num_features, activation, two_layers)

        # --- Initialize feature normalization layer if needed ---
        if self.normalize_features:
            self.normalization = nn.LayerNorm(normalized_shape=num_features)

    def get_features(self, state, keep_patches=False):
        """
        Extract feature vector from raw state using the encoder.
        Optionally normalizes the features.
        """
        with torch.no_grad():  # Disable gradient computation for efficiency
            x = self.encoder(state)
            if self.normalize_features:
                x = self.normalization(x)
        return x

    def get_value_from_features(self, features):
        """
        Compute the state value (V) from features using the critic network.
        """
        return self.critic(features)

    def get_probabilities_from_features(self, features):
        """
        Compute action probabilities (policy) from features using the actor network.
        """
        return self.actor(features)

    def get_value_from_state(self, state):
        """
        Compute the state value (V) directly from raw state.
        """
        return self.get_value_from_features(self.get_features(state))

    def get_probabilities_from_state(self, state):
        """
        Compute action probabilities (policy) directly from raw state.
        """
        return self.get_probabilities_from_features(self.get_features(state))

    def get_action_and_log_prob_from_features(self, features):
        """
        Sample an action from the policy and return its log probability.
        """
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_log_probs_entropy_from_features(self, features, action):
        """
        Compute the log probability and entropy of a given action.
        """
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.log_prob(action), dist.entropy()

    def get_action_and_log_prob_dist_from_features(self, features):
        """
        Sample an action, return its log probability, and the full distribution.
        """
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist

class A_Agent(AC_Agent):
    """
    Actor-only Agent (no critic).
    This agent only uses the actor network to select actions, without evaluating them.
    """
    def __init__(
        self,
        num_features,
        num_action,
        activation,
        encoder,
        normalize_features=False,
        *args,
        **kwargs
    ):
        # --- Initialize the parent class with have_critic=False ---
        super().__init__(
            num_features,
            num_action,
            activation,
            encoder,
            normalize_features,
            False,  # No critic
            *args,
            **kwargs
        )

class Discrete_Model_Based_Agent():
    """
    Discrete Model-Based Agent using Q-learning.
    This agent learns a Q-table and uses a discrete world model to plan and update Q-values.
    """

    def __init__(self, num_states, num_actions, encoder, epsilon, alpha, gamma):
        # --- Parameters ---
        # num_states: number of discrete states in the environment
        # num_actions: number of possible actions
        # encoder: neural network that encodes raw observations into discrete states
        # epsilon: exploration rate (epsilon-greedy policy)
        # alpha: learning rate for Q-value updates
        # gamma: discount factor for future rewards

        self.num_states = num_states
        self.num_actions = num_actions
        self.world_model = Discrete_Maze_Model(num_states, num_actions)  # Discrete world model
        self.encoder = encoder  # Encodes raw observations into discrete states
        self.epsilon = epsilon  # Exploration rate (can be a schedule)
        self.qvalues = torch.zeros((num_states, num_actions))  # Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def val(self, state, action):
        """
        Get the Q-value for a state-action pair.
        """
        return self.qvalues[state, action]

    def max_val(self, state):
        """
        Get the maximum Q-value for a state (used for bootstrapping).
        """
        return max([self.qvalues[state, a] for a in range(self.num_actions)])

    def get_features(self, obs):
        """
        Encode raw observation into a discrete state.
        """
        return self.encoder(obs).cpu()

    def update_q(self, state, action, new_state, reward):
        """
        Update the Q-value for a state-action pair using the Q-learning update rule.
        """
        max_next_s = self.max_val(new_state)
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
        self.qvalues[state, action] += self.alpha * (
            reward + self.gamma * max_next_s - self.val(state, action)
        )

    def get_action_from_state(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        eps = random.random()
        if eps < self.epsilon.get_lr():  # Explore: random action
            action = random.choice(range(self.num_actions))
        else:  # Exploit: best action according to Q-values
            curr = self.qvalues[state, 0]
            action = 0
            for a in range(self.num_actions):
                if self.qvalues[state, a] > curr:
                    action = a
                    curr = self.qvalues[state, a]
        return action