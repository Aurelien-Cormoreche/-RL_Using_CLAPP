import torch
import tqdm
from RL_algorithms.trainer_utils import get_features_from_state_encoder
from RL_algorithms.actor_critic.train import createschedulers, advantage_function, update_eligibility
from RL_algorithms.agents import AC_Agent
from utils.utils_torch import TorchDeque, InfoNceLoss, CustomAdamDuoEligibility
from RL_algorithms.dynamic_encoders import CLAPP_Layer, Encoding_Layer, Predictive_Encoding_Trainer, Contrastive_Encoding_Trainer
import torch.nn as nn
import random
def run_separate_dynamic_encoder(opt, envs, encoder, feature_dim, num_epochs, action_dim):

    agent = AC_Agent(feature_dim, action_dim,None, encoder, opt.normalize_features,two_layers= opt.two_layers).to(opt.device)
    actor = agent.actor
    critic = agent.critic
    critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler = createschedulers(opt)
    optimizer = CustomAdamDuoEligibility(actor, critic, opt.device, critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, opt.entropy, entropy_coeff_scheduler, opt.gamma)
    schedulders = [critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler]

    encoder_trainer = None
    if opt.encoder_layer == 'contrastive':
        encoder_layer_time = Encoding_Layer(feature_dim, opt.encoder_latent_dim_time)
        encoder_layer_time = encoder_layer_time.to(opt.device)
        loss_time = InfoNceLoss()
        num_negatives_time = 5
        encoder_trainer_time = Contrastive_Encoding_Trainer(opt, loss_time,encoder_layer_time, [5, 30, 2000], feature_dim, 1, num_negatives_time, True)

        encoder_layer_direction = Encoding_Layer(feature_dim, opt.encoder_latent_dim_direction)
        encoder_layer_direction = encoder_layer_direction.to(opt.device)
        loss_direction = InfoNceLoss()
        num_negatives_dir = 10
        encoder_trainer_direction = Contrastive_Encoding_Trainer(opt, loss_direction,encoder_layer_time, 10 , feature_dim, 1, num_negatives_dir, False)

    
    for epoch in tqdm.tqdm(range(num_epochs)):
        state, _ = envs.reset()
        features = get_features_from_state_encoder(opt, state, encoder, opt.device)
        memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= feature_dim, device= opt.device, dtype= torch.float32)
        memory.fill(features)
        done = False
        direction = torch.zeros((), dtype= torch.int, device= opt.device)
        length_episode = 0
        tot_encoding_loss_direction = 0
        tot_encoding_loss_time = 0
        num_updates_time = 0
        num_updates_direction = 0
        if opt.encoder_layer == 'contrastive':
            encoder_trainer_time.reset_memory()
            encoder_trainer_direction.reset_memory()
        optimizer.reset_zw_ztheta()
        tot_reward = 0
        while not done:
            if epoch > 60:
                if opt.encoder_layer == 'contrastive':
                    current_features = memory.get_all_content_as_tensor()
                    encoder_trainer_time.cascade_memory.push(current_features)
                    encoder_trainer_direction.cascade_memory.push(current_features, direction)
                    if encoder_trainer_time.cascade_memory.can_sample(num_negatives_time):
                        tot_encoding_loss_time += encoder_trainer_time.train_one_step(2, 10)
                        num_updates_time += 1
                    if encoder_trainer_direction.cascade_memory.can_sample(max(2,num_negatives_dir)):
                        tot_encoding_loss_direction += encoder_trainer_direction.train_one_step(2, 10)
                        num_updates_direction += 1
            
            action, logprob, dist = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())
            value = agent.get_value_from_features(memory.get_all_content_as_tensor())
            action_i = action.detach().item()
            if action_i == 0:
                direction = (direction + 1) % 8
            elif action_i == 1:
                direction = (direction - 1) % 8
            reward = 0
            for _ in range(opt.frame_skip):
                n_state, rewards, terminated, truncated, _ = envs.step([action_i])
                reward += rewards[0]
                length_episode += 1
                if terminated or truncated:
                    break
            tot_reward += reward            
            terminated = terminated[0]
            truncated = truncated[0]
        
            features = get_features_from_state_encoder(opt, n_state, encoder, opt.device)
            memory.push(features)
            
            advantage, delayed_value = advantage_function(reward, value, terminated, truncated, agent, memory, opt.target, None, opt.gamma)
            update_eligibility(value, advantage, logprob, dist.entropy(), optimizer)            
            
            done= terminated or truncated 
            
            if opt.render:
                envs.render()
        print(tot_reward)
        for scheduler in schedulders: scheduler.step_forward()
        if num_updates_time > 0:
            print(f'loss time: {tot_encoding_loss_time/num_updates_time}')
        if num_updates_direction > 0:
            print(f'loss direction: {tot_encoding_loss_direction/ num_updates_direction}')
        if epoch % 3 == 0:
            torch.save(encoder_layer_direction.state_dict(), 'trained_models/direction_contrastive_encoder.pt')
            torch.save(encoder_layer_time.state_dict(), 'trained_models/time_contrastive_encoder.pt')
        
        
        
    