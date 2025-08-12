import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch.optim.adamw import AdamW
from utils.utils_torch import Cascade_Memory

class Encoding_Trainer():
    def __init__(self, opt, model):
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), opt.encoder_lr)
        self.tot_loss = 0

    
    def compute_representation(self, x):
        self.predicted = self.model.predict_from_features(x)
        return self.predicted
    
    def updateEncoder(self, zero_out_predictions = True):
        self.optimizer.zero_grad()
        self.tot_loss.backward()
        self.optimizer.step()
        if zero_out_predictions:
            self.zero_out_predictions()

    def zero_out_predictions(self):
        self.tot_loss = 0

class Predictive_Encoding_Trainer(Encoding_Trainer):
    def __init__(self, opt, loss_function, model):
        super().__init__(opt, loss_function, model)
        self.loss_function = loss_function        
        self.predicted = None


    def compute_loss(self,real):
        loss = self.loss_function(self.predicted, real)     
        self.tot_loss += loss
        return loss.detach().item()
    

class Contrastive_Encoding_Trainer(Encoding_Trainer):
    def __init__(self, opt, loss_function, model, buffer_sizes, num_features, num_samples_pos, num_samples_neg):
        super().__init__(opt, model)
        self.buffer_sizes = buffer_sizes
        self.loss_function = loss_function
        self.cascade_memory = Cascade_Memory(buffer_sizes, num_features,  opt.device)
        self.num_samples_pos = num_samples_pos
        self.num_samples_neg = num_samples_neg
        self.tot_loss = 0

    def compute_loss(self, real):
        positives = self.model(self.cascade_memory.sample_recent(self.num_samples_pos))
        negatives = self.model(self.cascade_memory.sample_old(self.num_samples_neg))
        encoded_real = self.model(real)
        loss = self.loss_function(encoded_real, positives, negatives)
        self.tot_loss += loss
        return loss.detach().item()
    
    def reset_memory(self):
        self.cascade_memory.reset()

class CLAPP_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(input_dim)
        self.w = Linear(input_dim, hidden_dim)
        self.pred_w = Linear(hidden_dim, pred_dim)
        self.activation = ReLU()
        
    def forward(self, x):
        return self.activation(self.w(self.norm(x)))
    
    def predict_from_features(self, x):
        return self.predict_from_encoding(self.forward(x))
    
    def predict_from_encoding(self, e):
        return self.pred_w(e)

class Encoding_Layer(nn.Module):

    def __init__(self, feature_dim, output_dim, num_direction_modified_dims,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim))
        self.num_direction_modified_dims = num_direction_modified_dims
        self.feature_dim = feature_dim
    
    def forward(self, x):
        features, direction = torch.split(x, [self.feature_dim, 1] , dim= -1)
        angle = direction * torch.pi / 8
        representation = self.layer(features)
        toRotate = representation[..., : self.num_direction_modified_dims]
        toRotate = toRotate.reshape(toRotate.shape[0], -1, 2)
        rotated = torch.cat((toRotate[..., 0] * torch.cos(angle) -  toRotate[..., 1] * torch.sin(angle),
                                toRotate[..., 1] * torch.cos(angle) +  toRotate[..., 0] * torch.sin(angle)), dim= -1)

        return torch.cat(( representation[...,  self.num_direction_modified_dims :], rotated), dim= -1)
    






        
        
        



    
        
    
