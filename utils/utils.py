import torch
import math
import mlflow
import argparse
import os
import numpy as np 
import gymnasium as gym
from sklearn.decomposition import PCA
from mlflow import MlflowClient, MlflowException

from utils.load_standalone_model import load_model
from utils.tmaze_discretizer import TmazeDiscretizer


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
    parser.add_argument('--seed', default= 1, type= int, help= 'manual seed for training')
    parser.add_argument('--checkpoint_interval', default= 1000, type= int, help= 'interval at which to save the model weights')

    #hyperparameters for the training
    parser.add_argument('--num_epochs', default= 80000, type= int, help= 'number of epochs for the training')
    parser.add_argument('--gamma', default= 0.995, help= 'gamma for training in the environment')    
    parser.add_argument('--nb_stacked_frames', default= 1, type= int, help= 'number of stacked frames given as input')
    parser.add_argument('--frame_skip', default= 1, type= int, help= 'number of frames to skip')
    parser.add_argument('--use_ICM', action= 'store_true', help= 'wether to use intrisic curiosity module or not')
    parser.add_argument('--icm_lr', default= 1e-4, type= float, help= 'learning rate for the models of the ICM')
    parser.add_argument('--ICM_latent_dim', default= 128, type= int, help= 'latent dimension for ICM')
    parser.add_argument('--alpha_intrinsic_reward', default= 1e-1, type= float, help= 'intrisic reward coefficient')
    parser.add_argument('--num_updates_ICM', default= 1, type= int, help= 'number of updates for the ICM models')
    parser.add_argument('--PCA', action='store_true', help= 'use PCA for ICM')
    parser.add_argument('--lr_scheduler', action= 'store_true', help= 'add a lr scheduler')
    parser.add_argument('--normalize_features', action= 'store_true', help='normalize the features from the encoder')
    parser.add_argument('--target', default= True, type= bool, help='wether to use a target network')
    parser.add_argument('--tau', default= 0.05, type= float, help='by how much we update the taget network')

    parser.add_argument('--schedule_type_critic', default='linear', help='schedule type for the critic learning rate')
    parser.add_argument('--critic_lr_i', type=float, default=1e-4, help='initial learning rate for the critic')
    parser.add_argument('--critic_lr_e', type=float, default=1e-4, help='end learning rate for the critic')
    parser.add_argument('--critic_lr_m', type=float, default=1e-4, help='max critic learning rate (for warmup jobs)')
    parser.add_argument('--critic_len_w', type=int, default=10, help='warmup length for the critic learning rate scheduler')

    parser.add_argument('--schedule_type_actor', default='linear', help='schedule type for the actor learning rate')
    parser.add_argument('--actor_lr_i', type=float, default=9e-5, help='initial learning rate for the actor')
    parser.add_argument('--actor_lr_e', type=float, default=9e-5, help='end learning rate for the actor')
    parser.add_argument('--actor_lr_m', type=float, default=1e-4, help='max actor learning rate (for warmup jobs)')
    parser.add_argument('--actor_len_w', type=int, default=100, help='warmup length for the actor learning rate scheduler')

    parser.add_argument('--schedule_type_theta_lam', default='cosine_annealing', help='schedule type for the actor eligibility trace delay')
    parser.add_argument('--t_delay_theta_i', type=float, default=0.9, help='initial delay for actor in case of eligibility trace')
    parser.add_argument('--t_delay_theta_e', type=float, default=0.75, help='end delay for actor in case of eligibility trace')
    parser.add_argument('--theta_l_m', type=float, default=0.9, help='max actor eligibility trace delay (for warmup jobs)')
    parser.add_argument('--theta_l_len_w', type=int, default=10, help='warmup length for actor eligibility trace delay')

    parser.add_argument('--schedule_type_w_lam', default='linear', help='schedule type for the critic eligibility trace delay')
    parser.add_argument('--t_delay_w_i', type=float, default=0.9, help='initial delay for critic in case of eligibility trace')
    parser.add_argument('--t_delay_w_e', type=float, default=0.9, help='end delay for critic in case of eligibility trace')
    parser.add_argument('--w_l_m', type=float, default=0.9, help='max critic eligibility trace delay (for warmup jobs)')
    parser.add_argument('--w_l_len_w', type=int, default=10, help='warmup length for critic eligibility trace delay')

    parser.add_argument('--schedule_type_baseline', default='linear', help='schedule type for the baseline if we run reinforce with artificial baseline')
    parser.add_argument('--baseline_i', type=float, default=0.05, help='initial baseline ')
    parser.add_argument('--baseline_e', type=float, default=0.05, help='end baseline')

    parser.add_argument('--entropy', action='store_true', help='add an entropy component to the loss')
    parser.add_argument('--schedule_type_entropy', default='constant', help='schedule type for the critic coefficient')
    parser.add_argument('--coeff_entropy_i', type=float, default=0.0005, help='initial coefficient of the critic in the PPO loss')
    parser.add_argument('--coeff_entropy_e', type=float, default=0.005, help='end coefficient of the critic in the PPO loss')
    parser.add_argument('--coeff_entropy_m', type=float, default=0.005, help='max coefficient of the critic (for warmup schedule)')
    parser.add_argument('--coeff_entropy_len_w', type=int, default=2000, help='warmup length for the critic coefficient schedule')


    parser.add_argument('--len_rollout', default= 1024, type= int, help= 'length of the continuous rollout')
    parser.add_argument('--num_updates', default= 8, type= int, help= 'number of steps for the optimizer')
    parser.add_argument('--minibatch_size', default= 256, type= int,help= 'define minibatch size for offline learning')
    parser.add_argument('--lr', default= 5e-5, type= float, help='Lr in case we need only one learning rate for our algorithm')
    parser.add_argument('--lambda_gae', default= 0.97, type= float, help='Lamda used when calculating the GAE')
    parser.add_argument('--not_normalize_advantages', action= 'store_false', help= 'normalize the advantages of each minibatch')
    parser.add_argument('--critic_eps', default= 0.25, type= float,help= 'the epsilon for clipping the critic updates' )
    parser.add_argument('--actor_eps', default= 0.25, type= float, help= 'the epsilon for clipping the actor updates' )
    parser.add_argument('--coeff_critic', default= 0.5, type= float, help= 'coefficient of the critic in the PPO general loss' )
    parser.add_argument('--coeff_entropy', default= 0.0005, type= float, help= 'coefficient of the entropy in the PPO general loss' )
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
    torch.save(models_dict,f"{os.environ['SAVED_MODELS_S2025']}/saved_from_run.pt")


def create_ml_flow_experiment(experiment_name,uri =f"file:{os.environ['ML_RUNS_S2025']}"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)


def collect_and_store_features(args, filename, encoder, env):
    disc = TmazeDiscretizer(env, encoder)
    features = disc.extract_features_from_all_positions()
    np.save(filename, features)
    return features


def createPCA(args, filename, env, encoder, n_components):
    if os.path.exists(filename):
        features = np.load(filename)
    else:
        features = collect_and_store_features(args, filename, encoder, env)
    pca = PCA(n_components= n_components)
    pca.fit(features)
    return pca
    

