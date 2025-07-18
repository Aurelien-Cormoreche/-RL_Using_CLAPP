import torch
import torch.nn as nn
from .models import Predictor_Model
from torch.nn import Identity



class ICM(nn.Module):

    def __init__(self,  action_dim, features_dim, with_encoder ,encoded_features_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.featuresencoded_features_dim = encoded_features_dim
        self.predictor_model = Predictor_Model(action_dim,encoded_features_dim)
        if with_encoder:
            print('not implemented')
            raise NotImplementedError()
        else:
            assert features_dim == encoded_features_dim
            self.encoder_model = Identity()
    
    def forward(self, features_s, features_s_t1,action):
   
        one_hot_action = nn.functional.one_hot(action, num_classes=self.action_dim)


        phi_s = self.encoder_model(features_s)
        phi_s_t1 = self.encoder_model(features_s_t1)

        phi_hat_t1 = self.predictor_model(phi_s,one_hot_action)

        return phi_hat_t1, phi_s_t1


        

def update_ICM_predictor(predicted, real, icm_optimizer):
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted, real)
    icm_optimizer.zero_grad()
    loss.backward()
    icm_optimizer.step()
    return loss




