import os
import argparse
import miniworld.wrappers
import tqdm
import traceback

from tqdm import std
import miniworld
import gymnasium as gym

from RL_algorithms.actor_critic.train import train_actor_critic
from RL_algorithms.PPO.train import train_PPO

from utils.load_standalone_model import load_model
from utils.utils import save_models, create_ml_flow_experiment, parsing, create_envs, launch_experiment, createPCA

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import torch.nn as nn
import numpy as np
from torchsummary import summary
import mlflow

def train(opt, envs, model_path, device, models_dict):
    
    gamma = opt.gamma

    if opt.encoder == 'CLAPP':
        encoder = load_model(model_path= model_path).eval()
        feature_dim = 1024
        if not opt.greyscale:
            feature_dim *= 3
        if opt.keep_patches:
            feature_dim = 15 * 1024
    elif opt.encoder == 'resnet':    
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = 1000
    else:
        print('no available encoder matched the argument')
        return
    

    encoder = encoder.to(device)
    encoder.compile(backend="aot_eager")

    
    for param in encoder.parameters():
        param.requires_grad = False
    
    action_dim = envs.single_action_space.n
    feature_dim = feature_dim * opt.nb_stacked_frames


    if opt.track_run:
        mlflow.start_run(run_name= opt.run_name)
        mlflow.log_params(
                {
                    'algorithm' : opt.algorithm,
                    'num_envs' : opt.num_envs,
                    'greyscale' : opt.greyscale,
                    'encoder': opt.encoder,
                    'num_epochs': opt.num_epochs,
                    'gamma': gamma,
                    'keep_patches' : opt.keep_patches, 
                    'seed' : opt.seed,
                    'visible_reward' : opt.visible_reward                
                }
        )
    if opt.PCA:
        pca_module = createPCA(args, f'mlruns/encoded_features_{opt.encoder}', envs[0], encoder, opt.ICM_latent_dim)
    if opt.algorithm.startswith("actor_critic"):
        train_actor_critic(opt, envs, device, encoder, gamma, models_dict, True , action_dim,feature_dim, pca_module)
    else:
        train_PPO(opt, envs, device, encoder, gamma, models_dict, action_dim, feature_dim)
    envs.close()
 


def main(args):


   

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(args.seed)

    else:
        device = torch.device("cpu")
        print('cpu device no seed set')


    model_path = os.path.abspath('trained_models')

    models_dict = {}
   
    create_ml_flow_experiment(args.experiment_name)
    

    if args.experiment:

        run_dicts = [ 
            { 'run_name' : 'resnetTry',
              'algorithm' : 'PPO',
              'encoder' : 'resnet',
              'greyscale' : False
                },
                {
                'run_name' : 'CLAPPTry',
                'algorithm' : 'PPO',
                'encoder' : 'CLAPP' ,
                        
            }         
            
        ]

        seeds = [5,10]
        launch_experiment(args, run_dicts, seeds,args.experiment_name, device, models_dict)
    else:
        envs = create_envs(args, args.num_envs)
        train(opt= args, envs= envs,model_path= model_path,device =device, models_dict= models_dict)
    
if __name__ == '__main__':
    
    args = parsing()
    main(args)
