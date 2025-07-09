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
from utils.utils import save_models, create_ml_flow_experiment

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
        train_actor_critic(opt, env, device, encoder, gamma, models_dict, action_dim, CLAPP_FEATURE_DIM)

    env.close()
 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment',default='T_maze/custom_T_Maze_V0.py', help= 'name of the environment')
    parser.add_argument('--algorithm',default= 'actor_critic', help= 'type of RL algorithm to use')
    parser.add_argument('--encoder', default= "CLAPP", help="decide which encoder to use")
    parser.add_argument('--seed', default= 0, type= int, help= 'manual seed for training')
    parser.add_argument('--num_epochs', default= 1800, help= 'number of epochs for the training')
    parser.add_argument('--actor_lr', default= 1e-6, help= 'learning rate for the actor if the algorithm is actor critic')
    parser.add_argument('--critic_lr', default= 1e-6, help= 'learning rate for the critic if the algorithm is actor critic')
    parser.add_argument('--max_episode_steps', default= 800, help= 'max number of steps per environment')
    parser.add_argument('--gamma', default= 0.999, help= 'gamma for training in the environment')
    parser.add_argument('--track_run', default= False, help= 'track the training run with mlflow')
    parser.add_argument('--experiment_name', default= 'actor_critic_tMaze_default', help='name of experiment on mlFlow')
    parser.add_argument('--run_name', default= 'default_run', help= 'name of the run on MlFlow')
    parser.add_argument('--t_delay_theta', default= 0.9, help= 'delay for actor in case of eligibility trace')
    parser.add_argument('--t_delay_w', default= 0.9, help= 'delay for the critic in case of eligibility trace')

    args = parser.parse_args()
    
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    env = miniworld.wrappers.GreyscaleWrapper(gym.make("MyTMaze", max_episode_steps= args.max_episode_steps, render_mode = 'human'))
    env.render_frame = False
    

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")


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
