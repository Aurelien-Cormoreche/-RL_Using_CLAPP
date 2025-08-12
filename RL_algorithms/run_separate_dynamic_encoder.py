import torch
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
        encoder.add_module('addtional CLAPP layer', clapp_layer)
        encoder_trainer = Predictive_Encoding_Trainer(opt, nn.MSELoss(), clapp_layer)
    elif opt.encoder_layer == 'contrastive':
        encoder_layer = Encoding_Layer(feature_dim,opt.encoder_latent_dim, int(opt.encoder_latent_dim * 0.75))
        encoder_layer = encoder_layer.to('mps')
        encoder.add_module('additional encoder layer', encoder_layer)
        loss = InfoNceLoss()
        encoder_trainer = Contrastive_Encoding_Trainer(opt, loss,encoder_layer, [10, 20, 30], feature_dim + 1, 1, 5)
    
    for epoch in range(num_epochs):
        state, _ = envs.reset()
        features = get_features_from_state_encoder(opt, state, encoder, opt.device)
        memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= 1024, device= opt.device, dtype= torch.float32)
        memory.fill(features)
        
        done = False
        direction = torch.zeros((1), dtype= torch.float32, device= opt.device)
        length_episode = 0
        tot_encoding_loss = 0
        if opt.encoder_layer == 'contrastive':
            encoder_trainer.reset_memory()
        while not done:
            if opt.encoder_layer == 'predictive':
                current_features = memory.get_all_content_as_tensor()
                encoder_trainer.compute_representation(current_features)
                tot_encoding_loss += encoder_trainer.compute_loss(current_features)
                encoder_trainer.updateEncoder()
            elif opt.encoder_layer == 'contrastive':
                current_features = memory.get_all_content_as_tensor()
                current_features = torch.cat((current_features, direction), dim= -1).unsqueeze(0)
                encoder_trainer.cascade_memory.push(current_features)
                if encoder_trainer.cascade_memory.full():
                    tot_encoding_loss += encoder_trainer.compute_loss(current_features)
                    encoder_trainer.updateEncoder()
            action = random.randint(0, 2)

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
        print(tot_encoding_loss/length_episode)
        
        
    