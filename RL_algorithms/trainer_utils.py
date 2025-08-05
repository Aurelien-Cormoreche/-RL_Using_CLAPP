import torch

from utils.utils import save_models
from utils.utils_torch import CustomLrSchedulerLinear, CustomLrSchedulerCosineAnnealing, CustomWarmupCosineAnnealing


def get_features_from_state(opt,n_state, agent, device):
    n_state_t = torch.tensor(n_state, device= device, dtype= torch.float32)
    if opt.greyscale:
        n_state_t = torch.unsqueeze(n_state_t, dim= 1)
    else:
        n_state_t = n_state_t.reshape(n_state_t.shape[0], n_state_t.shape[3], n_state_t.shape[1], n_state_t.shape[2])
    features = agent.get_features(n_state_t).flatten()
    return features
    

def save_models_(opt, models_dict, agent, icm):
    if opt.algorithm != 'prioritized_sweeping':
        models_dict['actor'] = agent.actor.state_dict()
        models_dict['critic'] = agent.critic.state_dict()
        if opt.use_ICM:
            models_dict['icm_predictor'] = icm.predictor_model.state_dict()
        save_models(models_dict)


            
def update_target(target_critic, critic, tau):
    target_state_dict = target_critic.state_dict()
    critic_state_dict = critic.state_dict()
    for key in critic_state_dict:
        target_state_dict[key] = tau *critic_state_dict[key] + (1 - tau) * target_state_dict[key] 
    target_critic.load_state_dict(target_state_dict)


def defineScheduler(type, initial_lr, end_lr, num_epochs, max_lr = None, warmup_len = None):
    if type == 'linear':
        return CustomLrSchedulerLinear(initial_lr, end_lr, num_epochs) 
    if type == 'cosine_annealing':
        return CustomLrSchedulerCosineAnnealing(initial_lr, num_epochs, end_lr)
    if type == 'warmup_cosine_annealing':
        return CustomWarmupCosineAnnealing(initial_lr, max_lr, warmup_len, num_epochs, end_lr)
    else:
        print('constant scheduler')
        return CustomLrSchedulerLinear(initial_lr, initial_lr, num_epochs)  
         

