import torch
import torch.nn as nn
import random 
from .models import ActorModel, CriticModel, Discrete_Maze_Model

class AC_Agent(nn.Module):

    def __init__(self,num_features, num_action, activation, encoder,normalize_features = False, have_critic = True, two_layers = False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.normalize_features = normalize_features

        self.encoder = encoder

        self.actor = ActorModel(num_features, num_action, two_layers)
        if have_critic:
            self.critic = CriticModel(num_features, activation, two_layers)

        if self.normalize_features:
            self.normalization = nn.LayerNorm(normalized_shape= num_features)
        print(self.encoder)
        print(self.actor)
        print(self.critic)
    def get_features(self, state, keep_patches = False):
        with torch.no_grad():
            x = self.encoder(state) 
            if self.normalize_features:
                x = self.normalization(x)
        return x
    
    def get_value_from_features(self, features):
        return self.critic(features)
    
    def get_probabilities_from_features(self, features):
        return self.actor(features)
    
    def get_value_from_state(self, state):
        return self.get_value_from_features(self.get_features(state))
    
    def get_probabilities_from_state(self, state):
        return self.get_probabilities_from_features(self.get_features(state))
    

    def get_action_and_log_prob_from_features(self, features):
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs= probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def get_log_probs_entropy_from_features(self, features, action):
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs= probs)

        return dist.log_prob(action), dist.entropy()
    
    def get_action_and_log_prob_dist_from_features(self, features):
        probs = self.get_probabilities_from_features(features)
        dist = torch.distributions.Categorical(probs= probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist


    
class A_Agent(AC_Agent):
    def __init__(self, num_features, num_action, activation, encoder, normalize_features=False, *args, **kwargs):
        super().__init__(num_features, num_action, activation, encoder, normalize_features, False, *args, **kwargs)


    
    
class Discrete_Model_Based_Agent():
    def __init__(self, num_states, num_actions, encoder, epsilon, alpha, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.world_model = Discrete_Maze_Model(num_states, num_actions)
        self.encoder = encoder
        self.epsilon = epsilon
        self.qvalues = torch.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma

    def val(self, state, action):
        return self.qvalues[state, action]
    
    def max_val(self, state):
        return max([self.qvalues[state, a] for a in range(self.num_actions)])
    
    def get_features(self, obs):
        return self.encoder(obs).cpu()
    
    def update_q(self, state, action, new_state, reward):
        max_next_s = self.max_val(new_state)
        self.qvalues[state, action] += self.alpha * (reward + self.gamma * max_next_s - self.val(state, action))
    
    def get_action_from_state(self, state):
        eps = random.random()
        if eps < self.epsilon.get_lr():
            action =  random.choice(range(self.num_actions))
        else:
            curr = self.qvalues[state, 0]
            action = 0
            for a in range(self.num_actions):
                if self.qvalues[state, a] > curr:
                    action = a
        return action

                

        

    