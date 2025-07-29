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
from ..trainer_utils import update_target, get_features_from_state
from utils.utils import save_models
from utils.utils_torch import TorchDeque, CustomAdamEligibility, CustomLrSchedulerLinear, CustomLrSchedulerCosineAnnealing, CustomWarmupCosineAnnealing

def actor_critic_train(opt, envs, modules, variables, epoch):
    agent, icm, optimizer, icm_optimizer, target_critic, schedulders   = modules
    feature_dim, action_dim, eligibility_trace, _ , _ , _, _ = variables
    state, _ = envs.reset(seed = opt.seed + epoch * opt.seed)
    features = get_features_from_state(opt, state, agent, opt.device)
    memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= feature_dim, device= opt.device, dtype= torch.float32)
    memory.fill(features)
        
    done = False
    total_reward = 0
    length_episode = 0
    tot_loss_critic = 0
    tot_loss_actor = 0

    if eligibility_trace:
        optimizer.reset_zw_ztheta()

    while not done:   
        action, logprob, dist = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())
        value = agent.get_value_from_features(memory.get_all_content_as_tensor())
        entropy_dist = dist.entropy()
            
        for _ in range(opt.frame_skip):
            n_state, reward, terminated, truncated, _ = envs.step([action.detach().item()])
            length_episode += 1
            if terminated or truncated:
                break
                       
        reward = reward[0]
        terminated = terminated[0]
        truncated = truncated[0]
            
        old_features = features
        features = get_features_from_state(opt, n_state, agent, opt.device)
        memory.push(features)

        if opt.use_ICM:
            predicted, _ = icm(old_features,features, action)
            reward += opt.alpha_intrinsic_reward * update_ICM_predictor(predicted, features, icm_optimizer, icm.encoder_model, opt.device)
            for _ in range(opt.num_updates_ICM - 1):
                update_ICM_predictor(icm(old_features,features,action)[0], features, icm_optimizer, icm.encoder_model, opt.device)

        advantage, delayed_value = advantage_function(reward, value, terminated, truncated, agent, memory, opt.target, target_critic, opt.gamma)
            
        if not eligibility_trace:
            lc = loss_critic(value, delayed_value)
            la = loss_actor(logprob, logprob, advantage, opt.actor_eps)
            tot_loss = lc * opt.coeff_critic + la - dist.entropy() * opt.coeff_entropy
            loss_critic,loss_actor = update_a2c(tot_loss, optimizer)
            tot_loss_actor += loss_actor
            tot_loss_critic += loss_critic
        else:
            update_eligibility(value, advantage, logprob, entropy_dist, optimizer)
            
        if opt.target:
            update_target(target_critic, agent.critic, opt.tau) 
        
        total_reward += reward
        done= terminated or truncated 
        
        if opt.render:
            envs.render()
    
    for scheduler in schedulders: scheduler.step_forward()

    return feature_dim, action_dim, eligibility_trace, length_episode, total_reward, tot_loss_actor, tot_loss_critic

def actor_critic_metrics(opt, epoch, variables):
    _, _, _, length_episode, total_reward, tot_loss_actor, tot_loss_critic = variables
    mlflow.log_metrics( 
                {
                    'reward': total_reward,
                    'loss_actor': tot_loss_actor/length_episode,
                    'loss_critic':  tot_loss_critic/length_episode,
                    'length_episode': length_episode
                },
                step= epoch
            )
def actor_critic_init(opt, feature_dim, action_dim, envs):
    assert opt.num_envs == 1

    if opt.algorithm == "actor_critic_e":
        print("using eligibility traces")
        eligibility_traces = True
    else:
        print("not using eligibility traces")
        eligibility_traces = False
   
    return feature_dim, action_dim, eligibility_traces, 0, 0, 0, 0


def advantage_function(reward, value, terminated, truncated, agent, memory, target, target_critic, gamma):
    with torch.no_grad():
        if target:
            new_value = target_critic(memory.get_all_content_as_tensor()).detach() 
        else:
            new_value = agent.get_value_from_features(memory.get_all_content_as_tensor())
        if terminated or truncated:
            delayed_value = reward
        else:
            delayed_value = reward + gamma * new_value
        return delayed_value - value, delayed_value
            
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

def update_eligibility(value, advantage, logprob, entropy_dist, optimizer):
    optimizer.zero_grad()
    value.backward()
    logprob.backward(retain_graph = True)
    with torch.no_grad():
        optimizer.step(advantage, entropy_dist)
        

def actor_critic_log_params(opt):
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

def actor_critic_modules(opt, variables, encoder, models_dict):
    feature_dim, action_dim, eligibility_traces, _, _, _, _ = variables
    agent = AC_Agent(feature_dim, action_dim,None, encoder, opt.normalize_features).to(opt.device)
    actor = agent.actor
    critic = agent.critic
    
    icm = None
    icm_optimizer = None
    if opt.use_ICM:
        icm = ICM(action_dim, feature_dim, pca_module, opt.ICM_latent_dim, opt.device).to(opt.device)
        icm_optimizer =  torch.optim.AdamW(icm.parameters(), lr = opt.icm_lr)

    target_critic = None
    if opt.target:
        target_critic = CriticModel(feature_dim, None).to(opt.device)
        target_critic.load_state_dict(critic.state_dict())
        models_dict['target'] = target_critic

    if not eligibility_traces:
        optimizer = torch.optim.AdamW(agent.parameters(), lr = opt.lr)

    else:
        critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler = createschedulers(opt)
        optimizer = CustomAdamEligibility(actor, critic, opt.device, critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, opt.entropy, entropy_coeff_scheduler, opt.gamma)
        schedulders = [critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler]
    return agent,icm, optimizer,icm_optimizer, target_critic, schedulders
        
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
    



    
    