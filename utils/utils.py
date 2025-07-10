import torch
import math
import mlflow
import argparse
import miniworld
import os
import numpy as np 
import gymnasium as gym

from mlflow import MlflowClient, MlflowException

from utils.load_standalone_model import load_model



def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment',default='T_maze/custom_T_Maze_V0.py', help= 'name of the environment')
    parser.add_argument('--algorithm',default= 'actor_critic', help= 'type of RL algorithm to use')
    parser.add_argument('--encoder', default= "CLAPP", help="decide which encoder to use")
    parser.add_argument('--seed', default= 0, type= int, help= 'manual seed for training')
    parser.add_argument('--num_epochs', default= 1800, help= 'number of epochs for the training')
    parser.add_argument('--actor_lr', default= 5e-4, help= 'learning rate for the actor if the algorithm is actor critic')
    parser.add_argument('--critic_lr', default= 1e-4, help= 'learning rate for the critic if the algorithm is actor critic')
    parser.add_argument('--max_episode_steps', default= 400, help= 'max number of steps per environment')
    parser.add_argument('--gamma', default= 0.99, help= 'gamma for training in the environment')
    parser.add_argument('--track_run', action= 'store_true', help= 'track the training run with mlflow')
    parser.add_argument('--experiment_name', default= 'actor_critic_tMaze_default', help='name of experiment on mlFlow')
    parser.add_argument('--run_name', default= 'default_run', help= 'name of the run on MlFlow')
    parser.add_argument('--t_delay_theta', default= 0.9, help= 'delay for actor in case of eligibility trace')
    parser.add_argument('--t_delay_w', default= 0.9, help= 'delay for the critic in case of eligibility trace')
    parser.add_argument('--experiment', action= 'store_true', help= 'run a full scale MLflow experiment')
    parser.add_argument('--render', action= 'store_true', help= 'will render the maze')
    parser.add_argument('--keep_patches', action= 'store_true', help= 'keep the patches for the encoder')

    return parser.parse_args()
    
def create_env(args):
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    env = miniworld.wrappers.GreyscaleWrapper(gym.make("MyTMaze", max_episode_steps= args.max_episode_steps, render_mode = 'human' if args.render else None))
    env.render_frame = False
    

    return env
    
def launch_experiment(opt, run_dicts, seeds ,experiment_name, device, models_dict):

    from main import train
    
    create_ml_flow_experiment(experiment_name)

    model_path = os.path.abspath('trained_models')

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'mps':
            torch.mps.manual_seed(seed)
        elif device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        else:
            print('not possible to assign seed')
        for run_dict in run_dicts:
            env = create_env(opt)

            for key in run_dict:
                setattr(opt,key,run_dict[key])

            create_env(opt)
            train(opt, env, model_path,device, models_dict)
            


def save_models(models_dict):
    for name in models_dict:
        models_dict[name].to('cpu')
        models_dict[name] = models_dict[name].state_dict()
    
    torch.save(models_dict,'trained_models/saved_from_run.pt')



def create_ml_flow_experiment(experiment_name,uri ="file:mlruns"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)




'''
def get_wall_states(env):
    pos_list = [[1.37*(2*x+1)-0.22, 1.37, -1.37] for x in range(3)] \
                + [[1.37*(2*x+1)-0.22, 1.37, 1.37] for x in range(3)] \
                + [[10.74, 1.37, 1.37*(2*x+1)-6.85] for x in range(5)] \
                + [[8, 1.37, 1.37*(2*x+1)-6.85] for x in [0, 1, 3, 4]] \
                + [[9.37, 1.37, -6.85], [9.37, 1.37, 6.85]]
        
    dir_list = [-math.pi / 2 for _ in range(3)] \
                + [math.pi / 2 for _ in range(3)] \
                + [-math.pi for _ in range(5)] \
                + [0 for _ in range(4)] \
                + [-math.pi / 2 , math.pi / 2]
    
    for p, d in zip(pos_list, dir_list):
        return
              





def collect_features(env, model_path, device, all_layers = False):
    encoder = load_model(model_path= model_path).eval()
    encoder.to(device)

    if device.type == 'mps':
        encoder.compile(backend="aot_eager")
    else:
        encoder.compile()

    for param in encoder.parameters():
        param.requires_grad = False


    states = get_wall_states(env)

    features = encoder(states)

    return features


    '''




    

