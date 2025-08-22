import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch.optim.adamw import AdamW
from utils.utils_torch import Cascade_Direction_Memory, CascadeTime_Memory

# Base trainer for encoders
class Encoding_Trainer():
    """
    Base class for training an encoder model.

    Attributes:
        model (nn.Module): Encoder or predictive layer to train.
        optimizer (torch.optim.Optimizer): Optimizer for the model parameters.
        tot_loss (torch.Tensor or float): Accumulated loss over training steps.
    """
    def __init__(self, opt, model):
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), opt.encoder_lr)
        self.tot_loss = 0

    def compute_representation(self, x):
        """
        Forward pass through the encoder to compute feature representation.
        """
        self.predicted = self.model.predict_from_features(x)
        return self.predicted
    
    def updateEncoder(self, zero_out_predictions = True):
        """
        Performs a gradient step on the encoder using accumulated loss.
        """
        self.optimizer.zero_grad()
        self.tot_loss.backward()
        self.optimizer.step()
        if zero_out_predictions:
            self.zero_out_predictions()

    def zero_out_predictions(self):
        """
        Reset accumulated loss after an optimizer step.
        """
        self.tot_loss = 0

class Predictive_Encoding_Trainer(Encoding_Trainer):
    """
    Computes and accumulates loss between predicted and real encoded features.
    """
    def __init__(self, opt, loss_function, model):
        super().__init__(opt, loss_function, model)
        self.loss_function = loss_function        
        self.predicted = None


    def compute_loss(self,real):
        """
        Computes the loss and accumulates it in tot_loss.

        Args:
            real (torch.Tensor): Ground truth feature representation.

        Returns:
            float: Detached loss value for logging.
        """
        loss = self.loss_function(self.predicted, real)     
        self.tot_loss += loss
        return loss.detach().item()
    

class Contrastive_Encoding_Trainer(Encoding_Trainer):
    """
    Contrastive learning trainer using positive and negative samples.
    Supports time-based or directional cascade memory.
    """
    def __init__(self, opt, loss_function, model, buffer_sizes, num_features, num_samples_pos, num_samples_neg, time):
        super().__init__(opt, model)
        self.buffer_sizes = buffer_sizes
        self.loss_function = loss_function
        self.time = time
        # Initialize memory for positive/negative sampling
        if self.time:
            self.cascade_memory = CascadeTime_Memory(buffer_sizes, num_features,  opt.device)
        else:
            self.cascade_memory = Cascade_Direction_Memory(buffer_sizes, num_features, opt.device, opt.epsilon_i)
        self.num_samples_pos = num_samples_pos
        self.num_samples_neg = num_samples_neg
        self.tot_loss = 0


    def compute_loss(self, positives, negatives, sample):
        """
        Computes contrastive loss and accumulates it.
        """
        loss = self.loss_function(sample, positives, negatives)
        self.tot_loss += loss
        return loss.detach().item()
    
    def reset_memory(self):
        """
        Resets the cascade memory.
        """
        self.cascade_memory.reset()

    def train_one_step(self, num_epochs, batch_size):
        """
        Performs one training step for the contrastive encoder.
        Iterates over epochs and batch size to sample positives/negatives.
        """
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
    """
    Contrastive or predictive encoding layer.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        pred_dim (int): Prediction output dimension.
    """
    def __init__(self, input_dim, hidden_dim, pred_dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(input_dim)
        self.w = Linear(input_dim, hidden_dim)
        self.pred_w = Linear(hidden_dim, pred_dim)
        self.activation = ReLU()
        
    def forward(self, x):
        """
        Forward pass through normalization, linear, and activation layers.
        """
        return self.activation(self.w(self.norm(x)))
    
    def predict_from_features(self, x):
        """
        Predict output from raw features.
        """
        return self.predict_from_encoding(self.forward(x))
    
    def predict_from_encoding(self, e):
        """
        Predict output from encoded features.
        """
        return self.pred_w(e)

class Encoding_Layer(nn.Module):
    """
    Encoding layer with normalization, linear projection, and activation.
    """
    def __init__(self, feature_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim),
            nn.LeakyReLU(negative_slope= 0.2), 
            nn.LayerNorm(output_dim)
            )
        self.feature_dim = feature_dim
        
        # Xavier initialization for linear weights
        nn.init.xavier_uniform(self.layers[1].weight, gain = nn.init.calculate_gain('leaky_relu', 0.2))
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):

        out =  self.layers(x)
        if not torch.isfinite(x).all():
            print("Non-finite values in input!")
        return out

# Pretrained dynamic encoder combining multiple unique encoders
class Pretrained_Dynamic_Encoder(nn.Module):
    """
    Combines multiple encoders and optionally normalizes/concatenates features.

    Args:
        unique_encoders (list of nn.Module): List of encoder modules to combine.
        output_mode (str): 'replace' to use only new encoders output, 'concatenate' to combine with normalized input.
    """
    def __init__(self,unique_encoders, output_mode = 'replace', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique_encoders = nn.ModuleList(unique_encoders)
        self.normalization_features = nn.LayerNorm(unique_encoders[0].feature_dim)
        self.output_mode = output_mode

    def forward(self, x):
        # Pass input through all unique encoders
        outs = [m(x) for m in self.unique_encoders]
        outs = torch.cat(outs, dim = -1)
        if self.output_mode == 'replace':
            return outs
        if self.output_mode == 'concatenate':
            # Concatenate normalized input with encoded output
            return torch.cat((self.normalization_features(x), outs), dim= -1)
    






        
        
        



    
        
    
