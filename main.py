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
from utils.utils import save_models, create_ml_flow_experiment, parsing, create_env, launch_experiment, collect_features

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import mlflow

def train(opt, envs, model_path, device, models_dict):
    
    CLAPP_FEATURE_DIM = 1024
    if not opt.greyscale:
        CLAPP_FEATURE_DIM *= 3
    if opt.keep_patches:
        CLAPP_FEATURE_DIM = 15 * 1024
    gamma = opt.gamma

    if opt.encoder == 'CLAPP':
        encoder = load_model(model_path= model_path).eval()
    else:
        print('no available encoder matched the argument')
    
    encoder.to(device)
    
    if device.type == 'mps':
        encoder.compile(backend="aot_eager")
    else:
        encoder.compile()

    for param in encoder.parameters():
        param.requires_grad = False
    
    action_dim = envs.single_action_space.n


    if opt.algorithm.startswith("actor_critic"):
        if opt.track_run:
            mlflow.start_run(run_name= opt.run_name)
            mlflow.log_params(
                {
                    'num_envs' : opt.num_envs,
                    'greyscale' : opt.greyscale,
                    'lr1': opt.actor_lr,
                    'lr2': opt.critic_lr,
                    'encoder': opt.encoder,
                    'num_epochs': opt.num_epochs,
                    'gamma': gamma,
                    'keep_patches' : opt.keep_patches, 
                    'seed' : opt.seed                   
                }
        )
        train_actor_critic(opt, envs, device, encoder, gamma, models_dict, True , action_dim,CLAPP_FEATURE_DIM)
    else:
        if opt.track_run:
            mlflow.start_run(run_name= opt.run_name)
            mlflow.log_params(
                {
                    'num_envs' : opt.num_envs,
                    'greyscale' : opt.greyscale,
                    'lr' : opt.lt,
                    'encoder': opt.encoder,
                    'num_epochs': opt.num_epochs,
                    'gamma': gamma,
                    'lamda_GAE' : opt.lambda_gae,
                    'keep_patches' : opt.keep_patches,
                    'len_rollout' : opt.len_rollout,
                    'num_updates' : opt.num_updates,
                    'seed' : opt.seed

                }
        )
            train_PPO(opt, envs, device, encoder, gamma, models_dict, action_dim, CLAPP_FEATURE_DIM)
    envs.close()
 


def main(args):

    envs = create_env(args, args.num_envs)

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
    
    try:
        if args.experiment:

            run_dicts = [ 
                { 'run_name' : 'try1',
                    'actor_lr' : 1e-5,
                    'critic_lr' : 1e-3 },
                    {
                    'run_name' : 'try2',
                    'actor_lr' : 1e-5,
                    'critic_lr' : 1e-3 
                }         
            ]

            seeds = [1,5,10]
            launch_experiment(args, run_dicts, seeds, 'try', device, models_dict)
        else:
            train(opt= args, envs= envs,model_path= model_path,device =device, models_dict= models_dict)
    except Exception as e:
       print(e)
       print(traceback.format_exc())
       envs.close()
       #save_models(models_dict)

    save_models(models_dict)


    
if __name__ == '__main__':
    
    args = parsing()
    main(args)
