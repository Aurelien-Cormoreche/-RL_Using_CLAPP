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

def train_actor_critic(opt, env, device, encoder, gamma, models_dict, target, action_dim, clapp_feature_dim, tau = 0.05):

    assert env.num_envs == 1
    if opt.algorithm == "actor_critic_e":
        print("using eligibility traces")
        eligibility_traces = True
    else:
        print("not using eligibility traces")
        eligibility_traces = False
    
    agent = AC_Agent(clapp_feature_dim, action_dim,'LeakyReLU', encoder).to(device)

    actor = agent.actor
    critic = agent.critic

    if target:
        target_critic = CriticModel(clapp_feature_dim, 'LeakyReLU').to(device)
        target_critic.load_state_dict(critic.state_dict())
        models_dict['target'] = target_critic


    models_dict['agent'] = agent

    if not eligibility_traces:
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr = opt.actor_lr)
        critic_optimizer = torch.optim.AdamW(critic.parameters(),lr = opt.critic_lr)
    else:
        z_theta = [torch.zeros_like(p, device= device) for p in actor.parameters()]
        z_w = [torch.zeros_like(p, device= device) for p in critic.parameters()]
        t_delay_theta = opt.t_delay_theta
        t_delay_w = opt.t_delay_w
            
    current_rewards = 0

    step = torch.zeros([1], device= device)

    for epoch in tqdm.tqdm(range(opt.num_epochs)):
        
        state, _ = env.reset(seed = opt.seed + epoch)
     
        state = torch.tensor(state, device= device, dtype= torch.float32)
        if opt.greyscale:
            state = torch.unsqueeze(state, dim= 1)
    
        
        features = encoder(state, keep_patches = opt.keep_patches)
        features = features.flatten().unsqueeze(0)
       
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

            action, logprob = agent.get_action_and_log_prob_from_features(features)
            value = agent.get_value_from_features(features)

            n_state, reward, terminated, truncated, _ = env.step([action.detach().item()])

            reward = reward[0]
            terminated = terminated[0]
            truncated = truncated[0]

            n_state_t = torch.tensor(n_state, device= device, dtype= torch.float32)

            if opt.greyscale:
                n_state_t = torch.unsqueeze(n_state_t, dim= 1)
           
            features = agent.get_features(n_state_t).flatten().unsqueeze()
           
            if target:
                new_value = target_critic(features).detach() 
            else:
                new_value = agent.get_value_from_features(features)

            delayed_value = reward + gamma * new_value

            advantage = delayed_value - value if not done or truncated else reward
            
            if opt.track_run:
                mlflow.log_metric('values', value.detach().squeeze().item(),step= int(step.item()))
                mlflow.log_metric('advantage', advantage.detach().item(),step= int(step.item()))

            if not eligibility_traces:
                tot_loss_critic, tot_loss_actor = update_a2c(value, delayed_value, critic_optimizer, 
                                        advantage,logprob, actor_optimizer, tot_loss_critic, tot_loss_actor)
            
            else:
                update_eligibility(z_w, z_theta, t_delay_w, t_delay_theta, gamma, I,
                        value, advantage, logprob, critic, actor ,opt.critic_lr, opt.actor_lr)
            I = gamma * I

            if target:
                target_state_dict = target_critic.state_dict()
                critic_state_dict = critic.state_dict()
                for key in critic_state_dict:
                    target_state_dict[key] = tau *critic_state_dict[key] + (1 - tau) * target_state_dict[key] 
                target_critic.load_state_dict(target_state_dict)
            
            state = n_state_t
            total_reward += reward
            length_episode += 1
            step += 1
            done= terminated or truncated

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






     
def update_a2c(value, delayed_value, critic_optimizer, advantage, logprob, actor_optimizer, tot_loss_critic, tot_loss_actor):
    criterion_critic = nn.MSELoss()
    loss_critic = criterion_critic(delayed_value,value)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    tot_loss_critic += loss_critic.item()

    loss_actor = -logprob*advantage.detach()
    actor_optimizer.zero_grad()
    loss_actor.backward()

    actor_optimizer.step()
    tot_loss_actor += loss_actor.item()


    return(tot_loss_critic, tot_loss_actor)

def update_eligibility(z_w, z_theta, t_delay_w, t_delay_theta, gamma, I, value, advantage, logprob, critic, actor, lr_w, lr_theta):

    grad_values = torch.autograd.grad(value, critic.parameters())
    grad_policy = torch.autograd.grad(logprob, actor.parameters())
    
    z_w = [gamma * t_delay_w * z + I * p for z, p in zip(z_w, grad_values)]
    z_theta = [gamma * t_delay_theta * z + I * p for z, p in zip(z_theta, grad_policy)]

    with torch.no_grad():

        for p, z in zip(critic.parameters(), z_w):
            p  += lr_w * advantage.squeeze() * z

        for p, z in zip(actor.parameters(), z_theta):    
            p  += lr_theta * advantage.squeeze() * z
    
        
        
    


    
    