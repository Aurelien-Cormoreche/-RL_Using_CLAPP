import torch
import tqdm
from RL_algorithms.trainer_utils import get_features_from_state_encoder
from utils.utils_torch import TorchDeque, InfoNceLoss
from RL_algorithms.dynamic_encoders import CLAPP_Layer, Encoding_Layer, Predictive_Encoding_Trainer, Contrastive_Encoding_Trainer
import torch.nn as nn
import random
def run_separate_dynamic_encoder(opt, envs, encoder, feature_dim, num_epochs):

    encoder_trainer = None
    if opt.encoder_layer == 'predictive':
        clapp_layer = CLAPP_Layer(input_dim= feature_dim, hidden_dim= opt.encoder_latent_dim , pred_dim=feature_dim).to(opt.device).requires_grad_()
        clapp_layer = clapp_layer.to('mps')
        encoder_trainer = Predictive_Encoding_Trainer(opt, nn.MSELoss(), clapp_layer)
    elif opt.encoder_layer == 'contrastive':
        encoder_layer_time = Encoding_Layer(feature_dim, opt.encoder_latent_dim_time)
        encoder_layer_time = encoder_layer_time.to('mps')
        loss_time = InfoNceLoss()
        num_negatives_time = 5
        encoder_trainer_time = Contrastive_Encoding_Trainer(opt, loss_time,encoder_layer_time, [5, 30, 100], feature_dim, 1, num_negatives_time, True)

        encoder_layer_direction = Encoding_Layer(feature_dim, opt.encoder_latent_dim_direction)
        encoder_layer_direction = encoder_layer_direction.to('mps')
        loss_direction = InfoNceLoss()
        num_negatives_dir = 10
        encoder_trainer_direction = Contrastive_Encoding_Trainer(opt, loss_direction,encoder_layer_time, 10 , feature_dim, 1, num_negatives_dir, False)

    
    for epoch in range(num_epochs):
        state, _ = envs.reset()
        features = get_features_from_state_encoder(opt, state, encoder, opt.device)
        memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= 1024, device= opt.device, dtype= torch.float32)
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

        while not done:
            if opt.encoder_layer == 'predictive':
                current_features = memory.get_all_content_as_tensor()
                encoder_trainer.compute_representation(current_features)
                tot_encoding_loss += encoder_trainer.compute_loss(current_features)
                encoder_trainer.updateEncoder()
            elif opt.encoder_layer == 'contrastive':
                current_features = memory.get_all_content_as_tensor()
                encoder_trainer_time.cascade_memory.push(current_features)
                encoder_trainer_direction.cascade_memory.push(current_features, direction)
                if encoder_trainer_time.cascade_memory.full():
                    tot_encoding_loss_time += encoder_trainer_time.train_one_step(2, 10)
                    num_updates_time += 1
                if encoder_trainer_direction.cascade_memory.can_sample(max(2,num_negatives_dir)):
                    tot_encoding_loss_direction += encoder_trainer_direction.train_one_step(2, 10, direction)
                    num_updates_direction += 1
                   
            action = random.randint(0, 2)
            if action == 0:
                direction = (direction + 1) % 8
            elif action == 1:
                direction = (direction - 1) % 8
            for _ in range(opt.frame_skip):
                n_state, _, terminated, truncated, _ = envs.step([action])
                length_episode += 1
                if terminated or truncated:
                    break
                        
            terminated = terminated[0]
            truncated = truncated[0]
        
            features = get_features_from_state_encoder(opt, n_state, encoder, opt.device)
            memory.push(features)
            
            done= terminated or truncated 
            
            if opt.render:
                envs.render()
        if num_updates_time > 0:
            print(f'loss time: {tot_encoding_loss_time/num_updates_time}')
        if num_updates_direction > 0:
            print(f'loss direction: {tot_encoding_loss_direction/ num_updates_direction}')
        if epoch % 3 == 0:
            torch.save(encoder_layer_direction.state_dict(), 'trained_models/direction_contrastive_encoder.pt')
            torch.save(encoder_layer_time.state_dict(), 'trained_models/time_contrastive_encoder.pt')
        
        
    