import torch
import torch.nn as nn
from .models import Predictor_Model
from torch.nn import Identity



class ICM(nn.Module):

    def __init__(self,  action_dim, features_dim, encoder ,encoded_features_dim, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.featuresencoded_features_dim = encoded_features_dim
        self.predictor_model = Predictor_Model(action_dim,encoded_features_dim)
        self.device = device
        if not encoder:
            assert features_dim == encoded_features_dim
            self.encoder_model = Identity()
        else:
            self.encoder_model = encoder
    
    def forward(self, features_s, features_s_t1,action):
        with torch.no_grad():
            one_hot_action = nn.functional.one_hot(action, num_classes=self.action_dim).reshape(1, -1)
            
            phi_s = torch.tensor(self.encoder_model.transform(features_s.to('cpu').numpy().reshape(1, -1)), device= self.device)
            phi_s_t1 = torch.tensor(self.encoder_model.transform(features_s_t1.to('cpu').numpy().reshape(1, -1)), device= self.device)

        phi_hat_t1 = self.predictor_model(phi_s,one_hot_action)

        return phi_hat_t1, phi_s_t1


        

def update_ICM_predictor(predicted, real, icm_optimizer, encoder, device):
    real = torch.tensor(encoder.transform(real.to('cpu').numpy().reshape(1, -1)), device= device)
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted, real)
    icm_optimizer.zero_grad()
    loss.backward()
    icm_optimizer.step()
    return loss




