import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Import the CLAPP model loader (assuming the CLAPP code is in a file called clapp_model.py)
from utils.load_standalone_model import load_model as load_clapp_model

# Option 1: CLAPP as Feature Extractor for Actor-Critic
class ActCrit1Layer(nn.Module):
    def __init__(self, env, clapp_model_path, gamma=0.99, freeze_clapp=True):
        super().__init__()
        self.env = env
        self.gamma = gamma
        
        # Load CLAPP model
        self.clapp_model = load_clapp_model(clapp_model_path, option=0)
        
        # Freeze CLAPP parameters if using as feature extractor
        if freeze_clapp:
            for param in self.clapp_model.parameters():
                param.requires_grad = False
        
        # Determine feature dimension from CLAPP
        # CLAPP uses 27x27 patches, so we'll test with 92x92 (recommended size)
        with torch.no_grad():
            test_input = torch.randn(1, 1, 92, 92)  # Recommended input size
            test_features = self.clapp_model(test_input)
            feature_dim = test_features.shape[1]
        
        action_dim = env.action_space
        print(f"CLAPP feature dimension: {feature_dim}, Action dimension: {action_dim}")
        
        # Actor and Critic heads
        self.actor_fc = nn.Linear(feature_dim, action_dim)
        self.critic_fc = nn.Linear(feature_dim, 1)
        
        # For storing episode data
        self.rewards = []
        self.log_probs = []
        self.state_values = []
    
    
    def extract_features(self, state, keep_patches=False):
        state = torch.tensor(state, dtype=torch.float32)# Add batch dimension 
        state = state.view(state.shape[2], 1, state.shape[0], state.shape[1])  # Assuming state is (H, W, C)
        with torch.no_grad():
            features = self.clapp_model(state, all_layers=False, keep_patches=keep_patches)
        features = features[1] # Remove batch dimension if present
        return features
    
    def forward(self, state):
        features = self.extract_features(state)
        return self.actor_fc(features), self.critic_fc(features)
    
    def act(self, state):
        """Select an action using CLAPP features"""
        # Extract features using CLAPP
        action = None
        features = self.extract_features(state)
        
        
        # Get action probabilities and state value
        logits, state_value = self.actor_fc(features), self.critic_fc(features)
        
        action_probs = F.softmax(logits, dim=-1)
        action_probs = action_probs.squeeze()  # Remove batch dimension if present
        dist = torch.distributions.Categorical(action_probs)
      # Remove batch dimension if present
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store for training
        self.log_probs.append(log_prob)
        self.state_values.append(state_value.squeeze())
        
        return action.item()
    
    def clear_episode_data(self):
        self.rewards = []
        self.log_probs = []
        self.state_values = []
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def calculate_losses_and_update(self, optimizer):
        """Calculate losses and update model parameters"""
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Convert stored data to tensors
        log_probs = torch.stack(self.log_probs)
        state_values = torch.stack(self.state_values)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate advantages
        advantages = returns - state_values.detach()
        
        # Calculate losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        total_loss = actor_loss + critic_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
