import os

#from RL_algorithms.actor_critic.train import train_actor_critic
from RL_algorithms.PPO.train import train_PPO
from RL_algorithms.trainer import Trainer
from RL_algorithms.models import Encoder_Model
from utils.load_standalone_model import load_model
from utils.utils import save_models, create_ml_flow_experiment, parsing, create_envs, launch_experiment, createPCA, select_device
from spatial_representations.models import Spatial_Model
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import mlflow

def train(opt, envs, model_path, device, models_dict):
    
    gamma = opt.gamma

    encoder_models = []
    
    if opt.encoder.startswith('CLAPP'):
        encoder_models.append(load_model(model_path= model_path).eval())
        feature_dim = 1024
        if opt.keep_patches:
            feature_dim = 15 * 1024
    elif opt.encoder.startswith('resnet'):
        model_res = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model_res.fc = torch.nn.Identity()
        assert not opt.greyscale
        feature_dim = 2048
        encoder_models.append(model_res)
    
    if opt.encoder.endswith('one_hot'):
        one_hot_model = Spatial_Model(feature_dim, [32])
        one_hot_model.load_state_dict(torch.load('spatial_representations/one_hot/model.pt', map_location= device))
        feature_dim = 32
        encoder_models.append(one_hot_model)
        encoder_models.append(nn.Softmax(dim= -1))
        print('using one hot')
    
    encoder = Encoder_Model(encoder_models)
    encoder = encoder.to(device).requires_grad_(False)
    encoder.compile(backend="aot_eager")

    action_dim = envs.single_action_space.n
    feature_dim = feature_dim * opt.nb_stacked_frames

    trainer = Trainer(opt, envs, encoder, feature_dim, action_dim)
    trainer.train()

    envs.close()
 


def main(args):

   
    device = select_device(args)

    model_path = os.path.abspath('trained_models')

    models_dict = {}
   
    create_ml_flow_experiment(args.experiment_name)
    

    if args.experiment:

        run_dicts = [
            { 'run_name' : 'CLAPP_Normalized_2',
              'algorithm' : 'actor_critic_e',
              'encoder' : 'CLAPP',
              'greyscale' : True,
              'num_epochs' : 8000,
              'frame_skip' : 3,
              'num_envs' : 1,
              'actor_lr' : 5e-5,
              'critic_lr' : 1e-4,
              't_delay_theta' : 0.9,
              't_delay_w' : 0.9,
              'gamma' : 0.995,
              'normalize_features' : True,
            }
,               
        ]

        seeds = [5,10]
        launch_experiment(args, run_dicts, seeds,args.experiment_name, device, models_dict)
    else:
        envs = create_envs(args, args.num_envs)
        train(opt= args, envs= envs,model_path= model_path,device =device, models_dict= models_dict)
    
if __name__ == '__main__':
    args = parsing()
    main(args)
