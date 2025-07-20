import torch
import torch.nn as nn
import tqdm
import mlflow

import numpy as np
from tqdm import std
import miniworld
import gymnasium as gym

from ..ac_agent import AC_Agent
from ..models import CriticModel
from ..exploration_modules import ICM, update_ICM_predictor

from utils.utils import save_models
from utils.utils_torch import TorchDeque

import time

def train_actor_critic(opt, env, device, encoder, gamma, models_dict, target, action_dim, feature_dim, tau = 0.05):

    assert env.num_envs == 1
    
    if opt.track_run:
            mlflow.log_params(
                {
                    'actor_lr' : opt.actor_lr,
                    'critic_lr' : opt.critic_lr,
                }
            )

    if opt.algorithm == "actor_critic_e":
        print("using eligibility traces")
        eligibility_traces = True
        if opt.track_run:
                mlflow.log_params(
                    {
                        't_delay_theta' : opt.t_delay_theta,
                        't_delay_w' : opt.t_delay_w,
                    }
                )
    else:
        print("not using eligibility traces")
        eligibility_traces = False
  
    agent = AC_Agent(feature_dim, action_dim,None, encoder).to(device)
    
    
    if opt.use_ICM:
        icm = ICM(action_dim, feature_dim, False, feature_dim).to(device)
        icm_optimizer =  torch.optim.AdamW(icm.parameters(), lr = opt.icm_lr)


    actor = agent.actor
    critic = agent.critic

    if target:
        target_critic = CriticModel(feature_dim, None).to(device)
        target_critic.load_state_dict(critic.state_dict())
        models_dict['target'] = target_critic

    if not eligibility_traces:
        optimizer = torch.optim.AdamW(agent.parameters(), lr = opt.lr)
      
    else:
        z_theta = [torch.zeros_like(p, device= device) for p in actor.parameters()]
        z_w = [torch.zeros_like(p, device= device) for p in critic.parameters()]
        t_delay_theta = opt.t_delay_theta
        t_delay_w = opt.t_delay_w
            
    current_rewards = 0

    step = torch.zeros([1], device= device)

    for epoch in tqdm.tqdm(range(opt.num_epochs)):
        
        state, info = env.reset(seed = opt.seed + epoch)

        state = torch.tensor(state, device= device, dtype= torch.float32)
        if opt.greyscale:
            state = torch.unsqueeze(state, dim= 1)
    
        features = encoder(state, keep_patches = opt.keep_patches)
        features = features.flatten()

        memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= feature_dim, device= device, dtype= torch.float32)
        memory.fill(features)
        
        done = False
        total_reward = 0
        length_episode = 0
        tot_loss_critic = 0
        tot_loss_actor = 0
       
        if eligibility_traces:
            for z in z_theta: z.zero_()
            for z in z_w: z.zero_()
            I = 1

        while not done:
         
            
            action, logprob, dist = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())
            value = agent.get_value_from_features(memory.get_all_content_as_tensor())

            for _ in range(opt.frame_skip):
                n_state, reward, terminated, truncated, info = env.step([action.detach().item()])
                length_episode += 1

                if terminated or truncated:
                    break
            

            reward = reward[0]
            terminated = terminated[0]
            truncated = truncated[0]
            
            n_state_t = torch.tensor(n_state, device= device, dtype= torch.float32)

            if opt.greyscale:
                n_state_t = torch.unsqueeze(n_state_t, dim= 1)

            old_features = features

            features = agent.get_features(n_state_t).flatten()
            memory.push(features)

            if opt.use_ICM:
                predicted, _ = icm(old_features,None, action)
                reward += opt.alpha_intrinsic_reward * update_ICM_predictor(predicted, features, icm_optimizer)
                for _ in range(opt.num_updates_ICM - 1):
                    update_ICM_predictor(icm(old_features,None,action)[0], features, icm_optimizer)
               

            with torch.no_grad():
                if target:
                    new_value = target_critic(memory.get_all_content_as_tensor()).detach() 
                else:
                    new_value = agent.get_value_from_features(memory.get_all_content_as_tensor())

                if terminated or truncated:
                    delayed_value = reward
                else:
                    delayed_value = reward + gamma * new_value

                advantage = delayed_value - value

            if not eligibility_traces:

                lc = loss_critic(value, delayed_value)
                la = loss_actor(logprob, logprob, advantage, opt.actor_eps)
                tot_loss = lc * opt.coeff_critic + la - dist.entropy() * opt.coeff_entropy

                tot_loss_critic, tot_loss_actor = update_a2c(tot_loss, optimizer)
            
            else:
                z_theta, z_w = update_eligibility(z_w, z_theta, t_delay_w, t_delay_theta, gamma, I,
                        value, advantage, logprob, critic, actor ,opt.critic_lr, opt.actor_lr)
                I = gamma * I

            if target:
                update_target(target_critic, critic, tau)

            
            state = n_state_t
            total_reward += reward
            step += 1
            done= terminated or truncated

            if epoch % opt.checkpoint_interval == 0:
                models_dict['actor'] = actor.state_dict()
                models_dict['critic'] = critic.state_dict()
                if opt.use_ICM:
                    models_dict['icm_predictor'] = icm.predictor_model.state_dict()
                save_models(models_dict)

            if opt.render:
               env.render() 
            
        current_rewards += total_reward  

        if opt.track_run:
            mlflow.log_metrics(
                {
                    'reward': total_reward,
                    'loss_acotr': tot_loss_actor/length_episode,
                    'loss_critic':  tot_loss_critic/length_episode,
                    'length_episode': length_episode
                },
                step= epoch
            )
            


def update_target(target_critic, critic, tau):
    target_state_dict = target_critic.state_dict()
    critic_state_dict = critic.state_dict()
    for key in critic_state_dict:
        target_state_dict[key] = tau *critic_state_dict[key] + (1 - tau) * target_state_dict[key] 
    target_critic.load_state_dict(target_state_dict)


def loss_actor(log_prob_t, past_log_prob_t, advantage_t, epsilon_clipping):

    log_ratio_probs_t  = log_prob_t - past_log_prob_t
    ratio_probs_t = log_ratio_probs_t.exp()

    clipped_ratio_probs_t = torch.clamp(ratio_probs_t, 1 - epsilon_clipping, 1 + epsilon_clipping)

    loss = -torch.min(advantage_t * ratio_probs_t, advantage_t * clipped_ratio_probs_t)

    return loss
     
def loss_critic(value_t, delayed_value_t):
    return 0.5 * (value_t - delayed_value_t) ** 2

def update_a2c(tot_loss, optimizer):
    optimizer.zero_grad()
    tot_loss.backward()
    optimizer.step()
    


def update_eligibility(z_w, z_theta, t_delay_w, t_delay_theta, gamma, I, value, advantage, logprob, critic, actor, lr_w, lr_theta):
    
    grad_values = torch.autograd.grad(value, critic.parameters())
    grad_policy = torch.autograd.grad(logprob, actor.parameters())
    
    z_w = [gamma * t_delay_w * z +  p for z, p in zip(z_w, grad_values)]
    z_theta = [gamma * t_delay_theta * z +   p for z, p in zip(z_theta, grad_policy)]

    with torch.no_grad():

        for p, z in zip(critic.parameters(), z_w):
            p.add_(lr_w * advantage.squeeze() * z)
            

        for p, z in zip(actor.parameters(), z_theta):    
            p.add_(lr_theta * advantage.squeeze() * z)

    return z_theta, z_w
    
        
        
    


    
    