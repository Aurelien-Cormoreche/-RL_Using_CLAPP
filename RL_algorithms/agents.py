import torch
import torch.nn as nn

from .models import ActorModel, CriticModel


class AC_Agent(nn.Module):

    def __init__(self,num_features, num_action, activation, encoder,normalize_features = False, have_critic = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.normalize_features = normalize_features

        self.encoder = encoder

        self.actor = ActorModel(num_features, num_action)
        if have_critic:
            self.critic = CriticModel(num_features, activation)

        if self.normalize_features:
            self.normalization = nn.LayerNorm(normalized_shape= num_features)

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


    
    
    
    
        

    