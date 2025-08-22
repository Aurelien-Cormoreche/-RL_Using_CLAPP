import torch
import torch.nn as nn
from .models import Predictor_Model
from torch.nn import Identity



class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for reinforcement learning.

    Purpose:
        - Predicts the encoded features of the next state given the current state and action.
        - Provides intrinsic rewards to encourage exploration based on prediction error.

    Args:
        action_dim (int): Dimension of the action space.
        features_dim (int): Dimension of the raw input features.
        encoder (nn.Module or None): Encoder model to transform states into feature space.
                                     If None, Identity() is used.
        encoded_features_dim (int): Dimension of the encoded feature space.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    def __init__(self,  action_dim, features_dim, encoder ,encoded_features_dim, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.featuresencoded_features_dim = encoded_features_dim
        # Predictor model: predicts next state features from current features + action
        self.predictor_model = Predictor_Model(action_dim,encoded_features_dim)
        self.device = device
         # Use encoder if provided, otherwise Identity (no transformation)
        if not encoder:
            assert features_dim == encoded_features_dim
            self.encoder_model = Identity()
        else:
            self.encoder_model = encoder
    
    def forward(self, features_s, features_s_t1,action):
        """
        Forward pass of the ICM.

        Args:
            features_s (torch.Tensor): Features of the current state.
            features_s_t1 (torch.Tensor): Features of the next state.
            action (torch.Tensor): Action taken in the current state.

        Returns:
            phi_hat_t1 (torch.Tensor): Predicted next-state features.
            phi_s_t1 (torch.Tensor): Actual next-state features (encoded).
        """
        with torch.no_grad():
            # Convert action to one-hot encoding
            one_hot_action = nn.functional.one_hot(action, num_classes=self.action_dim).reshape(1, -1)
            
            # Encode current and next states using encoder (transform to numpy first)
            phi_s = torch.tensor(self.encoder_model.transform(features_s.to('cpu').numpy().reshape(1, -1)), device= self.device)
            phi_s_t1 = torch.tensor(self.encoder_model.transform(features_s_t1.to('cpu').numpy().reshape(1, -1)), device= self.device)
        
        # Predict next state features from current state and action
        phi_hat_t1 = self.predictor_model(phi_s,one_hot_action)

        return phi_hat_t1, phi_s_t1




def update_ICM_predictor(predicted, real, icm_optimizer, encoder, device):
    """
    Updates the ICM predictor network using the Smooth L1 loss between predicted
    and real encoded features of the next state.

    Args:
        predicted (torch.Tensor): Predicted features from ICM predictor.
        real (torch.Tensor): Real features of the next state.
        icm_optimizer (torch.optim.Optimizer): Optimizer for the ICM predictor.
        encoder (nn.Module): Encoder used to transform the real features.
        device (torch.device): Device to perform computation on.

    Returns:
        torch.Tensor: Loss value after the update step.
    """
    # Encode the real next state features
    real = torch.tensor(encoder.transform(real.to('cpu').numpy().reshape(1, -1)), device= device)
    
    # Use Smooth L1 Loss (Huber loss) for stability
    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted, real)
    
    # Backpropagation and optimizer step
    icm_optimizer.zero_grad()
    loss.backward()
    icm_optimizer.step()
    return loss




