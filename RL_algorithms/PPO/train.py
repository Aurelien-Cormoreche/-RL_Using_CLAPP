import torch
import torch.nn as nn
import tqdm
import mlflow

import numpy as np
from tqdm import std

from ..ac_agent import AC_Agent
from utils.utils import save_models


def ppo_log_params(opt):
    mlflow.log_params(
    {
        'lr' : opt.lr,                    
        'num_epochs': opt.num_epochs,
        'gamma': opt.gamma,
        'lamda_GAE' : opt.lambda_gae,
        'len_rollout' : opt.len_rollout,
        'num_updates' : opt.num_updates,
        'minibatch_size' : opt.minibatch_size,
        'not_normalize_advantages' : opt.not_normalize_advantages,
        'critic_eps' : opt.critic_eps,
        'actor_eps' : opt.actor_eps,
        'coeff_critic' : opt.coeff_critic,
        'coeff_entropy' : opt.coeff_entropy,
        'grad_clipping' : opt.grad_clipping
        }
    )

def ppo_metrics(epoch, variables):
    _, _, _, _, _, _, tot_loss_actor, tot_loss_critic, tot_loss, tot_entropy, samples_num, update, num_updates = variables
    mlflow.log_metrics(
        {
        'avg_loss_actor_batch' : tot_loss_actor/ samples_num,
        'avg_loss_critic_batch' : tot_loss_critic/ samples_num,
        'avg_loss_batch' : tot_loss / samples_num,
        'avg_entropy_batch' : tot_entropy / samples_num
        },
        step= update + epoch * num_updates          
    )      

def ppo_init(opt, feature_dim, action_dim, envs):

    states, _ = envs.reset(seed = opt.seed)
    states_t = torch.as_tensor(states, dtype= torch.float32, device= opt.device)
    if opt.greyscale:
        states_t = torch.unsqueeze(states_t, dim= 1)
    else:
        states_t = torch.reshape(states_t, (states_t.shape[0], states_t.shape[3], states_t.shape[1], states_t.shape[2]))

    is_next_observation_terminal_t = torch.zeros(opt.num_envs, device= opt.device)
    count_num_steps_env = torch.zeros((opt.num_envs,1), dtype= torch.float32, device= opt.device)
    nums_run = 0
    return feature_dim, action_dim, states_t, is_next_observation_terminal_t, count_num_steps_env, nums_run, 0, 0, 0, 0, 0, 0, 0

def ppo_modules(opt, variables, encoder, models_dict):
    feature_dim = variables[0]
    action_dim = variables[1]
    agent = AC_Agent(feature_dim, action_dim, None, encoder, opt.normalize_features).to(opt.device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr = opt.lr)
    return agent, None, optimizer


def train_PPO(opt, envs, device, encoder, gamma, models_dict, action_dim, feature_dim):

    if opt.track_run :
        mlflow.log_params(
            {
                'lr' : opt.lr,                    
                'num_epochs': opt.num_epochs,
                'gamma': gamma,
                'lamda_GAE' : opt.lambda_gae,
                'len_rollout' : opt.len_rollout,
                'num_updates' : opt.num_updates,
                'minibatch_size' : opt.minibatch_size,
                'not_normalize_advantages' : opt.not_normalize_advantages,
                'critic_eps' : opt.critic_eps,
                'actor_eps' : opt.actor_eps,
                'coeff_critic' : opt.coeff_critic,
                'coeff_entropy' : opt.coeff_entropy,
                'grad_clipping' : opt.grad_clipping
                }
        )
            

    num_envs = opt.num_envs
    
    agent = AC_Agent(feature_dim, action_dim, None, encoder, opt.normalize_features).to(device)

    optimizer = torch.optim.AdamW(agent.parameters(), lr = opt.lr)

    states, _ = envs.reset(seed = opt.seed)

    states_t = torch.as_tensor(states, dtype= torch.float32, device= device)
    
    if opt.greyscale:
        states_t = torch.unsqueeze(states_t, dim= 1)
    else:
        states_t = torch.reshape(states_t, (states_t.shape[0], states_t.shape[3], states_t.shape[1], states_t.shape[2]))

    is_next_observation_terminal_t = torch.zeros(num_envs, device= device)

    count_num_steps_env = torch.zeros((num_envs,1), dtype= torch.float32, device= device)
    nums_run = 0

    for epoch in tqdm.tqdm(range(opt.num_epochs)):

        (batch_features, 
        batch_log_probs,
        batch_actions,
        batch_advantages,
        batch_returns,
        batch_values,
        states_t,
        is_next_observation_terminal_t,
        nums_run) = collect_rollouts(opt, envs, device, agent, opt.len_rollout,
                                                           feature_dim, action_dim, gamma, states_t, 
                                                           is_next_observation_terminal_t, count_num_steps_env, nums_run)
       

        update_agent(opt, opt.num_updates, opt.len_rollout, num_envs, agent, 
                    optimizer, batch_features, batch_advantages, batch_log_probs, 
                    batch_returns, batch_values, batch_actions, epoch)
       
        if epoch % opt.checkpoint_interval == 0:
            models_dict['actor'] = agent.actor.state_dict()
            models_dict['critic'] = agent.critic.state_dict()
            save_models(models_dict)
            

def ppo_collector(opt, envs, modules, variables):
        agent = modules[0]
        num_envs = opt.num_envs
        len_rollouts = opt.len_rollout
        feature_dim, action_dim, states_t, is_next_observation_terminal_t, count_num_steps_env, nums_run, _, _, _, _, _, _, _ = variables

        batch_features = torch.empty((len_rollouts, num_envs, feature_dim ), device= opt.device)
        batch_log_probs = torch.empty((len_rollouts, num_envs), device= opt.device)
        batch_actions = torch.empty((len_rollouts, num_envs), device= opt.device)
        batch_rewards = torch.empty((len_rollouts, num_envs), device= opt.device)
        batch_values = torch.empty((len_rollouts, num_envs), device= opt.device)
        batch_is_episode_terminated = torch.empty((len_rollouts, num_envs), device= opt.device)

        features_t = agent.get_features(states_t)
        is_next_observation_terminal_t = torch.zeros(num_envs, device= opt.device)

        for step in range(len_rollouts):
            
            batch_features[step] = features_t
            batch_is_episode_terminated[step] = is_next_observation_terminal_t

            with torch.no_grad():

                actions_t, log_probs_from_actions_t = agent.get_action_and_log_prob_from_features(features_t)
                values_t = agent.get_value_from_features(features_t).squeeze()


            batch_actions[step] = actions_t
            batch_log_probs[step] = log_probs_from_actions_t

            batch_values[step] = values_t

              
            for _ in range(opt.frame_skip):
                count_num_steps_env += torch.ones_like(count_num_steps_env, dtype= torch.float32, device= opt.device)
                n_state, rewards, terminated, truncated, _ = envs.step(actions_t.cpu().numpy())
                if terminated or truncated:
                    break

            batch_rewards[step] = torch.as_tensor(rewards,dtype= torch.float32, device= opt.device)
            
            is_next_observation_terminal = np.logical_or(terminated, truncated)
            is_next_observation_terminal_t = torch.as_tensor(is_next_observation_terminal, dtype= torch.float32, device= opt.device)

            if opt.track_run:
                terminal_mask = is_next_observation_terminal_t == 1.0
                for elem in count_num_steps_env[terminal_mask]:
                    nums_run += 1
                    mlflow.log_metric('run length', elem.item(), step= nums_run)
                count_num_steps_env[terminal_mask] = 0.0
                

            states_t = torch.as_tensor(n_state, dtype= torch.float32, device= opt.device)
            if opt.greyscale:
                states_t = torch.unsqueeze(states_t, dim= 1)
            else:
                states_t = torch.reshape(states_t, (states_t.shape[0], states_t.shape[3], states_t.shape[1], states_t.shape[2]))
            
            features_t = agent.get_features(states_t)

            if opt.render:
                envs.render()

        with torch.no_grad():
            next_values_t = agent.get_value_from_features(features_t).squeeze()

            batch_advantages, batch_returns = compute_advantages(len_rollouts, num_envs, opt.gamma, opt.lambda_gae, opt.device, 
                                              is_next_observation_terminal_t, next_values_t, batch_is_episode_terminated,
                                                batch_values, batch_rewards)
            
        return  (batch_features.flatten(end_dim= -2), 
        batch_log_probs.flatten(),
        batch_actions.flatten(),
        batch_advantages.flatten(),
        batch_returns.flatten(),
        batch_values.flatten(),
        states_t,
        is_next_observation_terminal_t,
        nums_run)

'''    
def collect_rollouts(opt, envs, device, agent, len_rollouts, feature_dim, action_dim, gamma, n_states_t, is_next_observation_terminal_t, count_num_steps_env, nums_run):
    
        num_envs = opt.num_envs

        batch_features = torch.empty((len_rollouts, num_envs, feature_dim ), device= device)
        batch_log_probs = torch.empty((len_rollouts, num_envs), device= device)
        batch_actions = torch.empty((len_rollouts, num_envs), device= device)
        batch_rewards = torch.empty((len_rollouts, num_envs), device= device)
        batch_values = torch.empty((len_rollouts, num_envs), device= device)
        batch_is_episode_terminated = torch.empty((len_rollouts, num_envs), device= device)

        states_t = n_states_t
        features_t = agent.get_features(states_t)
        is_next_observation_terminal_t = torch.zeros(num_envs, device= device)

        for step in range(len_rollouts):
            
            batch_features[step] = features_t
            batch_is_episode_terminated[step] = is_next_observation_terminal_t

            with torch.no_grad():

                actions_t, log_probs_from_actions_t = agent.get_action_and_log_prob_from_features(features_t)
                values_t = agent.get_value_from_features(features_t).squeeze()


            batch_actions[step] = actions_t
            batch_log_probs[step] = log_probs_from_actions_t

            batch_values[step] = values_t

              
            for _ in range(opt.frame_skip):
                count_num_steps_env += torch.ones_like(count_num_steps_env, dtype= torch.float32, device= device)
                n_state, rewards, terminated, truncated, _ = envs.step(actions_t.cpu().numpy())
                if terminated or truncated:
                    break

            batch_rewards[step] = torch.as_tensor(rewards,dtype= torch.float32, device= device)
            
            is_next_observation_terminal = np.logical_or(terminated, truncated)
            is_next_observation_terminal_t = torch.as_tensor(is_next_observation_terminal, dtype= torch.float32, device= device)

            if opt.track_run:
                terminal_mask = is_next_observation_terminal_t == 1.0
                for elem in count_num_steps_env[terminal_mask]:
                    nums_run += 1
                    mlflow.log_metric('run length', elem.item(), step= nums_run)
                count_num_steps_env[terminal_mask] = 0.0
                

            states_t = torch.as_tensor(n_state, dtype= torch.float32, device= device)
            if opt.greyscale:
                states_t = torch.unsqueeze(states_t, dim= 1)
            else:
                states_t = torch.reshape(states_t, (states_t.shape[0], states_t.shape[3], states_t.shape[1], states_t.shape[2]))
            
            features_t = agent.get_features(states_t)

            if opt.render:
                envs.render()

        with torch.no_grad():
            next_values_t = agent.get_value_from_features(features_t).squeeze()

            batch_advantages, batch_returns = compute_advantages(len_rollouts, num_envs, gamma, opt.lambda_gae, device, 
                                              is_next_observation_terminal_t, next_values_t, batch_is_episode_terminated,
                                                batch_values, batch_rewards)
            
        return  (batch_features.flatten(end_dim= -2), 
        batch_log_probs.flatten(),
        batch_actions.flatten(),
        batch_advantages.flatten(),
        batch_returns.flatten(),
        batch_values.flatten(),
        states_t,
        is_next_observation_terminal_t,
        nums_run)

'''
def compute_advantages(len_rollouts, num_envs, gamma, lambda_gae, device, is_next_observation_terminal_t, next_value_t, is_episode_terminated_t, values_t, rewards):
     
    running_GAE = torch.zeros((num_envs), dtype= torch.float32, device= device)

    advantages_t = torch.empty((len_rollouts, num_envs), dtype= torch.float32, device= device)

    
    for t in reversed(range(len_rollouts)):
        if t == len_rollouts - 1:
               episode_continues_t = 1 - is_next_observation_terminal_t
               next_value_t = next_value_t
        else:
              episode_continues_t = 1 - is_episode_terminated_t[t + 1]
              next_value_t = values_t[t + 1]  
      
        td_error = rewards[t] + gamma * next_value_t * episode_continues_t - values_t[t]

        running_GAE = lambda_gae * gamma * running_GAE * episode_continues_t + td_error

        advantages_t[t] = running_GAE

    return advantages_t , advantages_t + values_t


def ppo_updator(opt, modules, variables, collected):

    feature_dim, action_dim, _, _, count_num_steps_env, _, _, _, _, _, _, _, _ = variables

    (batch_features, 
        batch_log_probs,
        batch_actions,
        batch_advantages,
        batch_returns,
        batch_values,
        states_t,
        is_next_observation_terminal_t,
        nums_run) = collected

    agent = modules[0]
    agent_optimizer = modules[2]

    samples_num = opt.len_rollout * opt.num_envs
    indices = np.arange(samples_num)

    tot_loss_actor = 0
    tot_loss_critic = 0
    tot_loss = 0
    tot_entropy = 0
     
    for update in range(opt.num_updates):
         
        np.random.shuffle(indices)

        for start in range(0, samples_num, opt.minibatch_size):
             
            minibatch_indices = indices[start : start + opt.minibatch_size]

            features_t = batch_features[minibatch_indices]

            actions_t = batch_actions[minibatch_indices]
            
            log_probs_t, entropies_t = agent.get_log_probs_entropy_from_features(features_t, actions_t)
            new_values_t = agent.get_value_from_features(features_t)

            advantages_t = batch_advantages[minibatch_indices]
  
            past_log_probs_t = batch_log_probs[minibatch_indices]
            returns_t = batch_returns[minibatch_indices]
            values_t = batch_values[minibatch_indices]

            loss_critic = compute_critic_loss(values_t, new_values_t, returns_t, opt.critic_eps)

            loss_actor = compute_actor_loss(opt, log_probs_t, past_log_probs_t, advantages_t, opt.actor_eps)

            loss_entropy = entropies_t.mean()

            loss = loss_critic * opt.coeff_critic + loss_actor - loss_entropy * opt.coeff_entropy

            agent_optimizer.zero_grad()
            loss.backward()

            if opt.grad_clipping:
                nn.utils.clip_grad_norm_(agent.parameters(),opt.max_grad_norm)
            agent_optimizer.step()

            tot_loss_actor += loss_actor
            tot_loss_critic += loss_critic
            tot_loss += loss
            tot_entropy += loss_entropy


    return  feature_dim, action_dim, states_t, is_next_observation_terminal_t, count_num_steps_env, nums_run, tot_loss_actor, tot_loss_critic, tot_loss, tot_entropy, samples_num, update, opt.num_updates

def update_agent(opt, num_updates, len_rollouts, num_envs, agent, agent_optimizer, batch_features,
                  batch_advantages, batch_log_probs, batch_returns, batch_values, batch_actions, epoch):

    samples_num = len_rollouts * num_envs
    indices = np.arange(samples_num)

    tot_loss_actor = 0
    tot_loss_critic = 0
    tot_loss = 0
    tot_entropy = 0
     
    for update in range(num_updates):
         
        np.random.shuffle(indices)

        for start in range(0, samples_num, opt.minibatch_size):
             
            minibatch_indices = indices[start : start + opt.minibatch_size]

            features_t = batch_features[minibatch_indices]

            actions_t = batch_actions[minibatch_indices]
            
            log_probs_t, entropies_t = agent.get_log_probs_entropy_from_features(features_t, actions_t)
            new_values_t = agent.get_value_from_features(features_t)

            advantages_t = batch_advantages[minibatch_indices]
  
            past_log_probs_t = batch_log_probs[minibatch_indices]
            returns_t = batch_returns[minibatch_indices]
            values_t = batch_values[minibatch_indices]

            loss_critic = compute_critic_loss(values_t, new_values_t, returns_t, opt.critic_eps)

            loss_actor = compute_actor_loss(opt, log_probs_t, past_log_probs_t, advantages_t, opt.actor_eps)

            loss_entropy = entropies_t.mean()

            loss = loss_critic * opt.coeff_critic + loss_actor - loss_entropy * opt.coeff_entropy

            agent_optimizer.zero_grad()
            loss.backward()

            if opt.grad_clipping:
                nn.utils.clip_grad_norm_(agent.parameters(),opt.max_grad_norm)
            agent_optimizer.step()

            tot_loss_actor += loss_actor
            tot_loss_critic += loss_critic
            tot_loss += loss
            tot_entropy += loss_entropy

  

def compute_critic_loss(value_t, new_value_t, return_t, epsilon_clipping, clipping = True):
    
    if clipping:
        loss_unclipped = (new_value_t - return_t) ** 2

        clipped_value_diff = torch.clamp(
              new_value_t - value_t,
              -epsilon_clipping,
              epsilon_clipping
          )
        
        clipped_value = (
              value_t + clipped_value_diff
          )

        clipped_loss = (clipped_value - return_t) ** 2

        v_loss_max = torch.max(loss_unclipped, clipped_loss)

        value_function_loss = 0.5 * v_loss_max.mean()

    else:
        value_function_loss = (
              0.5 * ((new_value_t - return_t) ** 2).mean()
          )
        
    return value_function_loss


def compute_actor_loss(opt, log_probs_t, past_log_probs_t, advantages_t, epsilon_clipping):

    log_ratio_probs_t  = log_probs_t - past_log_probs_t
    ratio_probs_t = log_ratio_probs_t.exp()
    opt.not_normalize_advantages
    if not   opt.not_normalize_advantages:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    clipped_ratio_probs_t = torch.clamp(ratio_probs_t, 1 - epsilon_clipping, 1 + epsilon_clipping)

    loss = -torch.min(advantages_t * ratio_probs_t, advantages_t * clipped_ratio_probs_t).mean()

    return loss
     
     


