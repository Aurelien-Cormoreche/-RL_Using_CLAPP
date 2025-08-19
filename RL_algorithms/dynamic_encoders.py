import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch.optim.adamw import AdamW
from utils.utils_torch import Cascade_Direction_Memory, CascadeTime_Memory

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
    def __init__(self, opt, loss_function, model, buffer_sizes, num_features, num_samples_pos, num_samples_neg, time):
        super().__init__(opt, model)
        self.buffer_sizes = buffer_sizes
        self.loss_function = loss_function
        self.time = time
        if self.time:
            self.cascade_memory = CascadeTime_Memory(buffer_sizes, num_features,  opt.device)
        else:
            self.cascade_memory = Cascade_Direction_Memory(buffer_sizes, num_features, opt.device, opt.epsilon_i)
        self.num_samples_pos = num_samples_pos
        self.num_samples_neg = num_samples_neg
        self.tot_loss = 0


    def compute_loss(self, positives, negatives, sample):
        loss = self.loss_function(sample, positives, negatives)
        self.tot_loss += loss
        return loss.detach().item()
    
    def reset_memory(self):
        self.cascade_memory.reset()

    def train_one_step(self, num_epochs, batch_size):
        direction = None
        loss = 0
        for e in range(num_epochs):
            for b in range(batch_size):
                if not self.time:
                    direction = torch.randint(high= 8, size= (1,))
                positives = self.model(self.cascade_memory.sample_posititves(self.num_samples_pos + 1, direction))
                negatives = self.model(self.cascade_memory.sample_negatives(self.num_samples_neg, direction))
                loss_b = self.compute_loss(positives[0].unsqueeze(0), negatives, positives[1])
                if e == 0 and b == 0:
                    loss += loss_b
            self.updateEncoder()
        return loss
                
            
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

    def __init__(self, feature_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim),
            nn.LeakyReLU(negative_slope= 0.2), 
            nn.LayerNorm(output_dim)
            )

        self.feature_dim = feature_dim
        nn.init.xavier_uniform(self.layers[1].weight, gain = nn.init.calculate_gain('leaky_relu', 0.2))
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):

        out =  self.layers(x)
        if not torch.isfinite(x).all():
            print("Non-finite values in input!")
        return out

class Pretrained_Dynamic_Encoder(nn.Module):
    def __init__(self,unique_encoders, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique_encoders = nn.ModuleList(unique_encoders)

    def forward(self, x):
        outs = [m(x) for m in self.unique_encoders]
        return torch.cat(outs, dim = -1)
    






        
        
        



    
        
    
