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
    """
    Run training of separate dynamic encoders (time-based and direction-based) 
    with an Actor-Critic agent in a reinforcement learning environment (the agent should normally be trained to explore the environment as much as possible).

    Args:
        opt: Configuration object containing training hyperparameters.
        envs: Vectorized RL environments.
        encoder: Pretrained encoder model.
        feature_dim (int): Dimensionality of feature representation from encoder.
        num_epochs (int): Number of training epochs.
        action_dim (int): Number of discrete actions in the environment.

    """
    # Initialize Actor-Critic agent
    agent = AC_Agent(feature_dim, action_dim,None, encoder, opt.normalize_features,two_layers= opt.two_layers).to(opt.device)
    actor = agent.actor
    critic = agent.critic
    
    # Create schedulers for learning rates and other parameters
    critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler = createschedulers(opt)
    # Custom optimizer for Actor-Critic with eligibility traces
    optimizer = CustomAdamDuoEligibility(actor, critic, opt.device, critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, opt.entropy, entropy_coeff_scheduler, opt.gamma)
    schedulders = [critic_lr_scheduler, actor_lr_scheduler, theta_lam_scheduler, w_lam_scheduler, entropy_coeff_scheduler]
    # Initialize dynamic encoder trainers for contrastive encoding if specified
    encoder_trainer = None
    if opt.encoder_layer == 'contrastive':
        # Time-based encoder
        encoder_layer_time = Encoding_Layer(feature_dim, opt.encoder_latent_dim_time)
        encoder_layer_time = encoder_layer_time.to(opt.device)
        loss_time = InfoNceLoss()
        num_negatives_time = 5
        encoder_trainer_time = Contrastive_Encoding_Trainer(opt, loss_time,encoder_layer_time, [20, 100, 1000], feature_dim, 1, num_negatives_time, True)
        # Direction-based encoder
        encoder_layer_direction = Encoding_Layer(feature_dim, opt.encoder_latent_dim_direction)
        encoder_layer_direction = encoder_layer_direction.to(opt.device)
        loss_direction = InfoNceLoss()
        num_negatives_dir = 10
        encoder_trainer_direction = Contrastive_Encoding_Trainer(opt, loss_direction,encoder_layer_time, 10 , feature_dim, 1, num_negatives_dir, False)

    # Main training loop
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
        optimizer.reset_zw_ztheta()  # Reset eligibility traces
        tot_reward = 0
        while not done:
            # Start training encoder after a certain number of epochs
            if epoch > 60:
                if opt.encoder_layer == 'contrastive':
                    current_features = memory.get_all_content_as_tensor()
                    # Push features to memory for contrastive sampling
                    encoder_trainer_time.cascade_memory.push(current_features)
                    encoder_trainer_direction.cascade_memory.push(current_features, direction)
                     # Train encoders if enough samples
                    if encoder_trainer_time.cascade_memory.can_sample(num_negatives_time):
                        tot_encoding_loss_time += encoder_trainer_time.train_one_step(2, 10)
                        num_updates_time += 1
                    if encoder_trainer_direction.cascade_memory.can_sample(max(2,num_negatives_dir)):
                        tot_encoding_loss_direction += encoder_trainer_direction.train_one_step(2, 10)
                        num_updates_direction += 1
            # Actor-Critic action selection and value estimation
            action, logprob, dist = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())
            value = agent.get_value_from_features(memory.get_all_content_as_tensor())
            action_i = action.detach().item()
            # Update direction for directional encoder
            if action_i == 0:
                direction = (direction + 1) % 8
            elif action_i == 1:
                direction = (direction - 1) % 8
            # Execute action in environment (with frame skipping)
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
            # Update memory with new state features
            features = get_features_from_state_encoder(opt, n_state, encoder, opt.device)
            memory.push(features)
             # Compute advantage and update Actor-Critic with eligibility traces
            advantage, delayed_value = advantage_function(reward, value, terminated, truncated, agent, memory, opt.target, None, opt.gamma)
            update_eligibility(value, advantage, logprob, dist.entropy(), optimizer)            
            
            done= terminated or truncated 
            
            if opt.render:
                envs.render()
         # Logging
        print(tot_reward)
        for scheduler in schedulders: scheduler.step_forward()
        if num_updates_time > 0:
            print(f'loss time: {tot_encoding_loss_time/num_updates_time}')
        if num_updates_direction > 0:
            print(f'loss direction: {tot_encoding_loss_direction/ num_updates_direction}')
        # Save encoder weights
        if epoch % 3 == 0:
            torch.save(encoder_layer_direction.state_dict(), 'trained_models/direction_contrastive_encoder.pt')
            torch.save(encoder_layer_time.state_dict(), 'trained_models/time_contrastive_encoder.pt')
        
        
        
    