import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, GELU, LeakyReLU, Softmax, Tanh, Identity
from collections import defaultdict
import tensorflow as tf
# Actor network for policy approximation
class ActorModel(nn.Module):
    def __init__(self, num_features, num_actions, two_layers = False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = 1024 # Hidden layer size if two_layers=True
        # Hidden layer size if two_layers=True
        if not two_layers:
                self.layer = nn.Sequential(Linear(num_features, num_actions))
        else:
            self.layer = nn.Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, num_actions))
        self.softmax = Softmax(dim= -1)  # Convert logits to probabilities

        # Initialize last layer to zeros for stable initial output (has proven to reduce convergence time by as much as 10)
        nn.init.zeros_(self.layer[-1].weight)
        nn.init.zeros_(self.layer[-1].bias)
        
    def forward(self, x, temp = None):
        # Optional temperature scaling
        if temp : 
            x = self.layer(x)/temp
        else:
           x = self.layer(x)

        x = self.softmax(x)
        return x
    
# Critic network for state-value approximation
class CriticModel(nn.Module):

    def __init__(self, num_features, activation = None, two_layers = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = 1024 # Hidden layer size if two_layers=True
        # Single-layer or two-layer architecture
        if not two_layers:
            self.layer = nn.Sequential(Linear(num_features, 1))
        else:
            self.layer = nn.Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1))
        
        # Initialize last layer to zeros for stable initial output (has proven to reduce convergence time by as much as 10)
        nn.init.zeros_(self.layer[-1].weight)
        nn.init.zeros_(self.layer[-1].bias)
    
        if activation == 'ReLu':
            self.activation = ReLU()
        if activation == 'GELU':
            self.activation = GELU()
        if activation == "LeakyReLU":
            self.activation = LeakyReLU()
        else:
            print('activation not found: continuing without')
            self.activation = Identity()
        
    def forward(self, x):
       return self.activation(self.layer(x))

# Predictor model: predicts next state features given current features and action used for ICM module
class Predictor_Model(nn.Module):

    def __init__(self, action_dim, encoded_features_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # Single linear layer combining encoded features and action
        self.layer = Linear(action_dim + encoded_features_dim, encoded_features_dim)

    def forward(self,encoded_features, action):
        # Concatenate features and action, then pass through linear layer
        return self.layer(torch.cat((encoded_features,action), dim= -1))
    
# Encoder model: stacks multiple models sequentially to form one big encoder
class Encoder_Model(nn.Module):

    def __init__(self,models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store models in a sequential container
        self.models = nn.Sequential(*models)
    
    def forward(self, x):
        # Pass input through all models
        x = self.models(x)
        return x
    

# Discrete tabular model for model-based RL in discrete state-action space (should be combined with a one hot encoder and priotititized sweeping)
class Discrete_Maze_Model():

    def __init__(self, num_states, num_actions):
        # Store predicted rewards for state-action pairs
        self.predicted_rewards = torch.zeros((num_states, num_actions))
        # Count times each action taken in each state
        self.times_action_taken_in_state = defaultdict(lambda : defaultdict(int))
        # Count times reaching new state from state-action pair
        self.times_state_from_state_action = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        # Track which state-action pairs lead to a given state
        self.states_pointing_to = [set() for _ in range(num_states)]
        self.num_actions = num_actions
        self.num_states = num_states


    def predicted_reward(self, state, action):
        # Return predicted reward for a state-action pair
        return self.predicted_rewards[state, action]
    
    def add(self, old_state, action, new_state, reward):
        # Register that (old_state, action) led to new_state with observed reward
        self.states_pointing_to[new_state].add((old_state, action))
        
        # Update counts
        self.times_action_taken_in_state[old_state][action] += 1
        self.times_state_from_state_action[old_state][action][new_state] += 1
        # Update predicted reward as running average
        self.predicted_rewards[old_state, action] = (self.predicted_rewards[old_state, action] * 
                                                     (self.times_action_taken_in_state[old_state][action] - 1) + reward
                                                     ) /  self.times_action_taken_in_state[old_state][action]

    
    def predict(self, state, action):
         # Predict next state probabilistically based on past counts
        dict_nums = self.times_state_from_state_action[state][action]
        tot_num = self.times_action_taken_in_state[state][action]
        probas_states = torch.tensor(list(dict_nums.values()))/ torch.tensor(tot_num)
        index_state = torch.distributions.Categorical(probas_states).sample()
        new_state = list(dict_nums.keys())[index_state]
        reward = self.predicted_reward(state, action)
        return new_state, reward
    
    def leading_to(self, state):
        # Return all state-action pairs that led to the given state
        return self.states_pointing_to[state]


class Keras_Encoder_Model(nn.Module):
    def __init__(self, keras_model, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.keras_model = keras_model

    @torch.no_grad()
    def forward(self, x):
        x_np = (
            x.detach()
             .cpu()
             .permute(0, 2, 3, 1)        
             .contiguous()
             .numpy()
             .astype("float32")
        )
        tf_out = self.kmodel(tf.convert_to_tensor(x_np), training=False)
        out_np = tf_out.numpy()
        return torch.from_numpy(out_np).to(x.device)
        


    

    

    
        
        