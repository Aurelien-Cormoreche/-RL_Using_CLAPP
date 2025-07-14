import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, GELU, LeakyReLU, Softmax, Tanh


class ActorModel(nn.Module):
    def __init__(self, num_features, num_actions,activation,*args, **kwargs):
        super().__init__(*args, **kwargs)
        

        self.layer = Linear(num_features, num_actions)
        
        if activation == 'ReLu':
            self.activation = ReLU()
        if activation == 'GELU':
            self.activation = GELU()
        if activation == "LeakyReLU":
            self.activation = LeakyReLU()
        if activation == "Tanh":
            self.activation == Tanh()
        else:
            print('activation not found: continuing with Tanh')
            self.activation = Tanh()

        self.softmax = Softmax(dim= 1)
        
    def forward(self, x, temp = None):
        if temp : 
            x = self.activation(self.layer(x))/temp
        else:
           x = self.activation(self.layer(x))
        x = self.softmax(x)
        return x
    
    

class CriticModel(nn.Module):

    def __init__(self, num_features, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Linear(num_features, 1)
        

        if activation == 'ReLu':
            self.activation = ReLU()
        if activation == 'GELU':
            self.activation = GELU()
        if activation == "LeakyReLU":
            self.activation = LeakyReLU()
        if activation == "Tanh":
            self.activation == Tanh()
        else:
            print('activation not found: continuing with Tanh')
            self.activation = Tanh()

        
    def forward(self, x):
       return self.activation(self.layer(x))








        