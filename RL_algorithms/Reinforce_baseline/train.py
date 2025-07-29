import torch
import torch.nn as nn
import tqdm
import mlflow
import numpy as np
from tqdm import std
import gymnasium as gym

from ..ac_agent import AC_Agent
from ..models import CriticModel
from ..exploration_modules import ICM, update_ICM_predictor

from utils.utils import save_models
from utils.utils_torch import TorchDeque, CustomAdamEligibility, CustomLrSchedulerLinear, CustomLrSchedulerCosineAnnealing, CustomWarmupCosineAnnealing

def train_reinforce_baseline(opt, env, device, encoder, gamma, models_dict, target, action_dim, feature_dim, pca_module = None, tau = 0.05):
    assert env.num_envs == 1 
    if opt.track_run:
        log_params(opt)
    
    agent, optimizer, icm, schedulders = createModules(opt, feature_dim, action_dim, encoder, 
                                                                       eligibility_traces, device, pca_module, target, models_dict, gamma)

    current_rewards = 0
    step = torch.zeros([1], device= device)

    for epoch in tqdm.tqdm(range(opt.num_epochs)):
        
        state, _ = env.reset(seed = opt.seed + epoch * opt.seed)
        features = get_features_from_state(opt, state, agent, device)
        memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= feature_dim, device= device, dtype= torch.float32)
        memory.fill(features)
        
        done = False
        total_reward = 0
        length_episode = 0

        optimizer.reset_zw_ztheta()

        while not done:   
            action, logprob, dist = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())
            
            for _ in range(opt.frame_skip):
                n_state, reward, terminated, truncated, _ = env.step([action.detach().item()])
                length_episode += 1
                if terminated or truncated:
                    break
            
            reward = reward[0]
            terminated = terminated[0]
            truncated = truncated[0]

            features = get_features_from_state(opt, n_state, agent, device)
            memory.push(features)
         
            total_reward += reward
            step += 1
            done= terminated or truncated
            
            for s in schedulders : 
                s.step_forward()

            if opt.render:
               env.render()       
        
        current_rewards += total_reward              
        if epoch % opt.checkpoint_interval == 0:
            save_models_(opt, models_dict, agent, icm)
        if opt.track_run:
            mlflow.log_metrics(
                {
                    'reward': total_reward,
                    'length_episode': length_episode
                },
                step= epoch
            )
            


def update_eligibility(value, advantage, logprob, entropy_dist, optimizer):
    optimizer.zero_grad()
    value.backward()
    logprob.backward(retain_graph = True)
    with torch.no_grad():
        optimizer.step(advantage, entropy_dist)
        
def save_models_(opt,models_dict, agent, icm):
    models_dict['actor'] = agent.actor.state_dict()
    models_dict['critic'] = agent.critic.state_dict()
    if opt.use_ICM:
        models_dict['icm_predictor'] = icm.predictor_model.state_dict()
    save_models(models_dict)

def log_params(opt):
    mlflow.log_params(
        {
        'actor_lr' : opt.actor_lr_i,
        'critic_lr' : opt.critic_lr_i,
    })

    if opt.algorithm == "actor_critic_e":
        mlflow.log_params({
            # Critic learning rate scheduler
            'schedule_type_critic': opt.schedule_type_critic,
            'critic_lr_i': opt.critic_lr_i,
            'critic_lr_e': opt.critic_lr_e,
            'critic_lr_m': opt.critic_lr_m,
            'critic_len_w': opt.critic_len_w,

            # Actor learning rate scheduler
            'schedule_type_actor': opt.schedule_type_actor,
            'actor_lr_i': opt.actor_lr_i,
            'actor_lr_e': opt.actor_lr_e,
            'actor_lr_m': opt.actor_lr_m,
            'actor_len_w': opt.actor_len_w,

            # Actor eligibility trace scheduler
            'schedule_type_theta_lam': opt.schedule_type_theta_lam,
            't_delay_theta_i': opt.t_delay_theta_i,
            't_delay_theta_e': opt.t_delay_theta_e,
            'theta_l_m': opt.theta_l_m,
            'theta_l_len_w': opt.theta_l_len_w,

            # Critic eligibility trace scheduler
            'schedule_type_w_lam': opt.schedule_type_w_lam,
            't_delay_w_i': opt.t_delay_w_i,
            't_delay_w_e': opt.t_delay_w_e,
            'w_l_m': opt.w_l_m,
            'w_l_len_w': opt.w_l_len_w,
        })

def createModules(opt, feature_dim, action_dim, encoder, eligibility_traces, device, pca_module, target, models_dict, gamma):
    agent = AC_Agent(feature_dim, action_dim,None, encoder, opt.normalize_features).to(device)
    actor = agent.actor


    critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler = createschedulers(opt)
    optimizer = CustomAdamEligibility(actor, critic, device, critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, opt.entropy, entropy_coeff_scheduler, gamma)
    schedulders = [critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler]
    return agent,optimizer, target_critic, schedulders
        
def createschedulers(opt):

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

    critic_lr_scheduler = defineScheduler(opt.schedule_type_critic, opt.critic_lr_i, opt.critic_lr_e, opt.num_epochs, opt.critic_lr_m, opt.critic_len_w)
    actor_lr_scheduler = defineScheduler(opt.schedule_type_actor, opt.actor_lr_i, opt.actor_lr_e, opt.num_epochs, opt.actor_lr_m, opt.actor_len_w)
    theta_lam_scheduler =defineScheduler(opt.schedule_type_theta_lam, opt.t_delay_theta_i, opt.t_delay_theta_e, opt.num_epochs, opt.theta_l_m, opt.theta_l_len_w)
    w_lam_scheduler = defineScheduler(opt.schedule_type_w_lam, opt.t_delay_w_i, opt.t_delay_w_e, opt.num_epochs, opt.w_l_m, opt.w_l_len_w)
    entropy_coeff_scheduler = defineScheduler(opt.schedule_type_entropy, opt.coeff_entropy_i, opt.coeff_entropy_e, opt.num_epochs, opt.coeff_entropy_m, opt.coeff_entropy_len_w)

    return critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler
    
def get_features_from_state(opt,n_state, agent, device):
    n_state_t = torch.tensor(n_state, device= device, dtype= torch.float32)
    if opt.greyscale:
        n_state_t = torch.unsqueeze(n_state_t, dim= 1)
    else:
        n_state_t = n_state_t.reshape(n_state_t.shape[0], n_state_t.shape[3], n_state_t.shape[1], n_state_t.shape[2])
    features = agent.get_features(n_state_t).flatten()
    return features
    


    
    