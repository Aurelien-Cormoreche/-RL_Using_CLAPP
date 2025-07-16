import miniworld.wrappers
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
    #arguments for the environment
    parser.add_argument('--environment',default='T_maze/custom_T_Maze_V0.py', help= 'name of the environment')
    parser.add_argument('--greyscale', action= 'store_true', help = 'determine if we keep render the state in greyscale')
    parser.add_argument('--render', action= 'store_true', help= 'will render the maze')
    parser.add_argument('--num_envs', type= int ,default= 8, help= 'the number of synchronous environment to spawn')
    parser.add_argument('--visible_reward', action= 'store_true', help= 'If the reward is a visible red box or not')
    parser.add_argument('--max_episode_steps', default= 1000, help= 'max number of steps per environment')
    #arguments for the training
    parser.add_argument('--algorithm',default= 'actor_critic', help= 'type of RL algorithm to use')
    parser.add_argument('--encoder', default= "CLAPP", help="decide which encoder to use")
    parser.add_argument('--keep_patches', action= 'store_true', help= 'keep the patches for the encoder')
    parser.add_argument('--seed', default= 0, type= int, help= 'manual seed for training')
    parser.add_argument('--checkpoint_interval', default= 50, type= int, help= 'interval at which to save the model weights')

    #hyperparameters for the training
    parser.add_argument('--num_epochs', default= 170, type= int, help= 'number of epochs for the training')
    parser.add_argument('--gamma', default= 0.995, help= 'gamma for training in the environment')    
        
    parser.add_argument('--actor_lr', default= 1e-3, help= 'learning rate for the actor if the algorithm is actor critic')
    parser.add_argument('--critic_lr', default= 5e-4, help= 'learning rate for the critic if the algorithm is actor critic')
    parser.add_argument('--t_delay_theta', default= 0.95, help= 'delay for actor in case of eligibility trace')
    parser.add_argument('--t_delay_w', default= 0.95, help= 'delay for the critic in case of eligibility trace')

    parser.add_argument('--len_rollout', default= 1024, type= int, help= 'length of the continuous rollout')
    parser.add_argument('--num_updates', default= 8, type= int, help= 'number of steps for the optimizer')
    parser.add_argument('--minibatch_size', default= 256, help= 'define minibatch size for offline learning')
    parser.add_argument('--lr', default= 5e-5, help='Lr in case we need only one learning rate for our algorithm')
    parser.add_argument('--lambda_gae', default= 0.97, help='Lamda used when calculating the GAE')
    parser.add_argument('--not_normalize_advantages', action= 'store_false', help= 'normalize the advantages of each minibatch')
    parser.add_argument('--critic_eps', default= 0.25, help= 'the epsilon for clipping the critic updates' )
    parser.add_argument('--actor_eps', default= 0.25, help= 'the epsilon for clipping the actor updates' )
    parser.add_argument('--coeff_critic', default= 0.5, help= 'coefficient of the critic in the PPO general loss' )
    parser.add_argument('--coeff_entropy', default= 0.0005, help= 'coefficient of the entropy in the PPO general loss' )
    parser.add_argument('--grad_clipping', action= 'store_true', help= 'do we need to clip the gradients' )

    #MlFlow parameters
    parser.add_argument('--track_run', action= 'store_true', help= 'track the training run with mlflow')
    parser.add_argument('--experiment_name', default= 'actor_critic_tMaze_default', help='name of experiment on mlFlow')
    parser.add_argument('--run_name', default= 'default_run', help= 'name of the run on MlFlow')
    parser.add_argument('--experiment', action= 'store_true', help= 'run a full scale MLflow experiment')

    return parser.parse_args()
    
def create_envs(args, num_envs):
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    envs =gym.make_vec("MyTMaze", num_envs= num_envs,  
                       max_episode_steps= args.max_episode_steps, render_mode = 'human' if args.render else None, visible_reward = args.visible_reward)

    if args.greyscale:
        envs = gym.wrappers.vector.GrayscaleObservation(envs)
    return envs
    
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

            for key in run_dict:
                setattr(opt,key,run_dict[key])
        
            env = create_envs(opt,opt.num_envs)
            train(opt, env, model_path,device, models_dict)
            mlflow.end_run()
            


def save_models(models_dict):
    
    torch.save(models_dict,'trained_models/saved_from_run.pt')



def create_ml_flow_experiment(experiment_name,uri ="file:mlruns"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)





def get_wall_states(env, pos_list, direction_list, device):
    
    states = []
    for p, d in zip(pos_list, direction_list):
        d = d * math.pi/180
    
        state, _= env.reset()
        env.unwrapped.agent.pos = p  # p = (x, 0, z)
        env.unwrapped.agent.dir = d 
        state = env.unwrapped.render_obs()
        env.render()
        state = torch.tensor(state, device= device, dtype= torch.float32)
        state = state.reshape(state.shape[2], state.shape[0], state.shape[1]) 

        states.append(state)

    states = torch.stack(states)
   
    return states


def collect_features(env, model_path, device, pos_list, direction_list, all_layers = False):
    encoder = load_model(model_path= model_path).eval()
    encoder.to(device)

    if device.type == 'mps':
        encoder.compile(backend="aot_eager")
    else:
        encoder.compile()

    for param in encoder.parameters():
        param.requires_grad = False

    states = get_wall_states(env, pos_list, direction_list, device)
    
    features = encoder(states)

    return features



    

