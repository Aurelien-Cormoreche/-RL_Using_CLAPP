import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, GELU, LeakyReLU, Softmax, Tanh, Identity


class ActorModel(nn.Module):
    def __init__(self, num_features, num_actions,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Linear(num_features, num_actions)
        self.softmax = Softmax(dim= -1)
        
    def forward(self, x, temp = None):
        if temp : 
            x = self.layer(x)/temp
        else:
           x = self.layer(x)


        x = self.softmax(x)
        return x
    
    

class CriticModel(nn.Module):

    def __init__(self, num_features, activation = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer = Linear(num_features, 1)
        
    
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
    
    
    