import torch

from utils.utils import save_models
from utils.utils_torch import CustomLrSchedulerLinear, CustomLrSchedulerCosineAnnealing, CustomWarmupCosineAnnealing


def get_features_from_state(opt,n_state, agent, device):
    """
    Extract features from a given state using the agent's feature extractor.

    Args:
        opt: Configuration object with attributes like 'greyscale'.
        n_state (np.array or torch.Tensor): The current state(s).
        agent: RL agent with a 'get_features' method.
        device: Torch device (cpu or cuda).

    Returns:
        torch.Tensor: Flattened feature vector for the state.
    """
    # Convert state to a torch tensor
    n_state_t = torch.tensor(n_state, device= device, dtype= torch.float32)

    if opt.greyscale:
        # For greyscale images, add a channel dimension
        n_state_t = torch.unsqueeze(n_state_t, dim= 1)
    else:
        # For RGB or multi-channel images, rearrange to (batch, channels, height, width)
        n_state_t = n_state_t.reshape(n_state_t.shape[0], n_state_t.shape[3], n_state_t.shape[1], n_state_t.shape[2])
    
    # Pass through agent's encoder and flatten
    features = agent.get_features(n_state_t).flatten()
    
    return features

def get_features_from_state_encoder(opt,n_state, encoder, device):
    """
    Extract features from a given state using a generic encoder model.

    Args:
        opt: Configuration object with attributes like 'greyscale'.
        n_state (np.array or torch.Tensor): The current state(s).
        encoder: Encoder neural network model (e.g., Encoder_Model).
        device: Torch device (cpu or cuda).

    Returns:
        torch.Tensor: Flattened feature vector for the state.
    """
    # Convert state to tensor
    n_state_t = torch.tensor(n_state, device= device, dtype= torch.float32)

    if opt.greyscale:
        n_state_t = torch.unsqueeze(n_state_t, dim= 1)
    else:
        n_state_t = n_state_t.reshape(n_state_t.shape[0], n_state_t.shape[3], n_state_t.shape[1], n_state_t.shape[2])
    
    # Pass through encoder and flatten
    features = encoder(n_state_t).flatten()
    
    return features

def save_models_(opt, models_dict, agent, icm):
    """
    Save the RL models (actor, critic, and optionally ICM predictor) to disk.

    Args:
        opt: Configuration object with algorithm and ICM settings.
        models_dict (dict): Dictionary storing model state_dicts.
        agent: RL agent with 'actor' and 'critic'.
        icm: Intrinsic Curiosity Module containing a predictor model.
    """
    if opt.algorithm != 'prioritized_sweeping':
        models_dict['actor'] = agent.actor.state_dict()
        models_dict['critic'] = agent.critic.state_dict()
        if opt.use_ICM:
            models_dict['icm_predictor'] = icm.predictor_model.state_dict()
        save_models(opt,models_dict)


            
def update_target(target_critic, critic, tau):
    """
    Soft-update target critic network parameters using Polyak averaging.

    Args:
        target_critic: Target critic network to update.
        critic: Current critic network.
        tau (float): Interpolation parameter between 0 and 1.
    """
    target_state_dict = target_critic.state_dict()
    critic_state_dict = critic.state_dict()
    # Update each parameter using tau
    for key in critic_state_dict:
        target_state_dict[key] = tau *critic_state_dict[key] + (1 - tau) * target_state_dict[key] 
    # Load updated parameters into target network
    target_critic.load_state_dict(target_state_dict)


def defineScheduler(type, initial_lr, end_lr, num_epochs, max_lr = None, warmup_len = None):
    """
    Create a learning rate scheduler based on the specified type.

    Args:
        type (str): Scheduler type ('linear', 'cosine_annealing', 'warmup_cosine_annealing').
        initial_lr (float): Starting learning rate.
        end_lr (float): Final learning rate.
        num_epochs (int): Total number of training epochs.
        max_lr (float, optional): Maximum LR for warmup cosine annealing.
        warmup_len (int, optional): Number of warmup steps for warmup cosine.

    Returns:
        Scheduler object that adjusts the learning rate during training.
    """
    if type == 'linear':
        return CustomLrSchedulerLinear(initial_lr, end_lr, num_epochs) 
    if type == 'cosine_annealing':
        return CustomLrSchedulerCosineAnnealing(initial_lr, num_epochs, end_lr)
    if type == 'warmup_cosine_annealing':
        return CustomWarmupCosineAnnealing(initial_lr, max_lr, warmup_len, num_epochs, end_lr)
    else:
        #default to a constant scheduler
        print('constant scheduler')
        return CustomLrSchedulerLinear(initial_lr, initial_lr, num_epochs)  
         

