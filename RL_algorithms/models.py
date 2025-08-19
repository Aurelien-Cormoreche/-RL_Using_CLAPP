import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, GELU, LeakyReLU, Softmax, Tanh, Identity
from collections import defaultdict

class ActorModel(nn.Module):
    def __init__(self, num_features, num_actions, two_layers = False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = 1024
        if not two_layers:
            self.layer = nn.Sequential(Linear(num_features, num_actions))
        else:
            self.layer = nn.Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, num_actions))
        self.softmax = Softmax(dim= -1)
        
        nn.init.zeros_(self.layer[-1].weight)
        nn.init.zeros_(self.layer[-1].bias)
        
    def forward(self, x, temp = None):
        if temp : 
            x = self.layer(x)/temp
        else:
           x = self.layer(x)

        x = self.softmax(x)
        return x
    
class CriticModel(nn.Module):

    def __init__(self, num_features, activation = None, two_layers = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = 1024
       
        if not two_layers:
            self.layer = nn.Sequential(Linear(num_features, 1))
        else:
            self.layer = nn.Sequential(
                Linear(num_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, 1))

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

class Predictor_Model(nn.Module):

    def __init__(self, action_dim, encoded_features_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Linear(action_dim + encoded_features_dim, encoded_features_dim)

    def forward(self,encoded_features, action):
        return self.layer(torch.cat((encoded_features,action), dim= -1))
    
    
class Encoder_Model(nn.Module):

    def __init__(self,models, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.models = nn.Sequential(*models)
    
    def forward(self, x):
        x = self.models(x)
        return x
    


class Discrete_Maze_Model():

    def __init__(self, num_states, num_actions):
        self.predicted_rewards = torch.zeros((num_states, num_actions))
        self.times_action_taken_in_state = defaultdict(lambda : defaultdict(int))
        self.times_state_from_state_action = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        self.states_pointing_to = [set() for _ in range(num_states)]
        self.num_actions = num_actions
        self.num_states = num_states


    def predicted_reward(self, state, action):
        return self.predicted_rewards[state, action]
    
    def add(self, old_state, action, new_state, reward):
        self.states_pointing_to[new_state].add((old_state, action))
    
        self.times_action_taken_in_state[old_state][action] += 1

        self.times_state_from_state_action[old_state][action][new_state] += 1

        self.predicted_rewards[old_state, action] = (self.predicted_rewards[old_state, action] * (self.times_action_taken_in_state[old_state][action] - 1) + reward) /  self.times_action_taken_in_state[old_state][action]

    
    def predict(self, state, action):
        dict_nums = self.times_state_from_state_action[state][action]
        tot_num = self.times_action_taken_in_state[state][action]
        probas_states = torch.tensor(list(dict_nums.values()))/ torch.tensor(tot_num)
        index_state = torch.distributions.Categorical(probas_states).sample()
        new_state = list(dict_nums.keys())[index_state]
        reward = self.predicted_reward(state, action)
        return new_state, reward
    
    def leading_to(self, state):
        return self.states_pointing_to[state]






        

        
        


    

    

    
        
        