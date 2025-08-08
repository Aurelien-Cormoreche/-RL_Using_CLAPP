import torch.nn as nn
from torch.nn import Linear, ReLU
from torch.optim.adamw import AdamW

class Predictive_Encoding_Trainer():
    def __init__(self, opt, loss_function, model):
        self.loss_function = loss_function
        self.model = model
        self.predicted = None
        self.tot_loss = 0
        self.optimizer = AdamW(self.model.parameters(), opt.encoder_lr)

    def compute_prediction_loss(self,real):
        loss = self.loss_function(self.predicted, real)     
        self.tot_loss += loss
        return loss.detach()

    def updateEncoder(self, zero_out_predictions = True):
        self.optimizer.zero_grad()
        self.tot_loss.backward()
        self.optimizer.step()
        if zero_out_predictions:
            self.zero_out_predictions()

    def zero_out_predictions(self):
        self.tot_loss = 0


class CLAPP_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = Linear(input_dim, hidden_dim)
        self.pred_w = Linear(hidden_dim, pred_dim)
        self.activation = ReLU()
        
    def forward(self, x):
        return self.activation(self.w(x))
    
    def predict_from_features(self, x):
        return self.predict_from_encoding(self.forward(x))
    
    def predict_from_encoding(self, e):
        return self.pred_w(e)
    
        
    
