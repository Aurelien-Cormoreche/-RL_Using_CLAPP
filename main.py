import os
import argparse
import miniworld.wrappers
import tqdm
import traceback

from tqdm import std
import miniworld
import gymnasium as gym

from RL_algorithms.actor_critic.train import train_actor_critic
from utils.load_standalone_model import load_model
from utils.utils import save_models, create_ml_flow_experiment, parsing, create_env

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import mlflow

def train(opt, env, model_path, device, models_dict):
    
    CLAPP_FEATURE_DIM = 1024
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
    
    action_dim = env.action_space.n

    if opt.track_run:
        mlflow.start_run(run_name= opt.run_name)
        mlflow.log_params(
            {
                'lr1': opt.actor_lr,
                'lr2': opt.critic_lr,
                'encoder': opt.encoder,
                'num_epochs': opt.num_epochs,
                'gamma': gamma
            }
        )

    if opt.algorithm.startswith("actor_critic"):
        train_actor_critic(opt, env, device, encoder, gamma, models_dict, True , action_dim,CLAPP_FEATURE_DIM)

    env.close()
 


def main():

    args = parsing()

    env = create_env(args)


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
        train(opt= args, env= env,model_path= model_path,device =device, models_dict= models_dict)
    except Exception as e:
       print(e)
       print(traceback.format_exc())
       #save_models(models_dict)

    save_models(models_dict)

    
    




    


    
if __name__ == '__main__':
    main()
