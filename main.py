import os
import argparse
import miniworld.wrappers
import tqdm

from tqdm import std
import miniworld
import gymnasium as gym



from RL_algorithms.actor_critic.models import ActorModel, CriticModel
from RL_algorithms.actor_critic.act_1layer_alg import ActCrit1Layer
from utils.load_standalone_model import load_model
from envs.T_maze.custom_T_Maze_V0 import MyTmaze
from utils.utils import save_model_and_state, create_ml_flow_experiment

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import mlflow

def train(opt, env, model_path, device):
    
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


    actor = ActorModel(CLAPP_FEATURE_DIM, action_dim).to(device)
    critic = CriticModel(CLAPP_FEATURE_DIM,'GELU').to(device)

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr = opt.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(),lr = opt.critic_lr)

    current_rewards = 0

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
    
    for epoch in tqdm.tqdm(range(opt.num_epochs)):
        
        state, info = env.reset()
        state = torch.tensor(state, device= device, dtype= torch.float32)
        state = state.reshape(state.shape[2], 1, state.shape[0], state.shape[1]) 
        features = encoder(state)

        done = False
        total_reward = 0
        length_episode = 0
        tot_loss_critic = 0
        tot_loss_actor = 0
        while not done:

            probs_action = actor(features)
            value = critic(features)

            dist = torch.distributions.Categorical(probs_action)
            action = dist.sample()
            

            n_state, reward, terminated, truncated, info = env.step(action)

            n_state = torch.tensor(n_state, device= device, dtype= torch.float32)
            n_state = state.reshape(n_state.shape[2], 1, n_state.shape[0], n_state.shape[1]) 
            
            features = encoder(n_state)
            delayed_value = reward + gamma * critic(features).detach()
            advantage = delayed_value - value

            criterion_critic = nn.SmoothL1Loss()
            loss_critic = criterion_critic(delayed_value,value)
            critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_optimizer.step()
            tot_loss_critic += loss_critic.detach()
            

            loss_actor = -dist.log_prob(action)*advantage.detach()
            actor_optimizer.zero_grad()
            loss_actor.backward()
            actor_optimizer.step()
            tot_loss_actor += loss_actor.detach()

            state = n_state
            total_reward += reward
            length_episode += 1
            done= terminated or truncated
        
        current_rewards += total_reward  

        if opt.track_run:
            mlflow.log_metrics(
                {
                    'reward': total_reward,
                    'loss_acotr': tot_loss_actor/length_episode,
                    'loss_critic':  tot_loss_critic/length_episode,
                    'length_episode': length_episode
                }
            )
            
        
        if epoch % 100 == 0:
            std.tqdm.write(f'Epoch number {epoch}, Average reward over the 100 last epochs: {current_rewards/100}')
            current_rewards = 0
    

    env.close()
 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment',default='T_maze/custom_T_Maze_V0.py', help= 'name of the environment')
    parser.add_argument('--algorithm',default= 'actor_critic', help= 'type of RL algorithm to use')
    parser.add_argument('--encoder', default= "CLAPP", help="decide which encoder to use")
    parser.add_argument('--seed', default= 0, help= 'manual seed for training')
    parser.add_argument('--num_epochs', default= 1800, help= 'number of epochs for the training')
    parser.add_argument('--actor_lr', default= 1e-2, help= 'learning rate for the actor if the algorithm is actor critic')
    parser.add_argument('--critic_lr', default= 1e-2, help= 'learning rate for the critic if the algorithm is actor critic')
    parser.add_argument('--max_episode_steps', default= 1500, help= 'max number of steps per environment')
    parser.add_argument('--gamma', default= 0.99, help= 'gamma for training in the environment')
    parser.add_argument('--track_run', default= False, help= 'track the training run with mlflow')
    parser.add_argument('--experiment_name', default= 'actor_critic_tMaze_default', help='name of experiment on mlFlow')
    parser.add_argument('--run_name', default= 'default_run', help= 'name of the run on MlFlow')

    args = parser.parse_args()
    
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    env = miniworld.wrappers.GreyscaleWrapper(gym.make("MyTMaze", render_mode="human", max_episode_steps= args.max_episode_steps))

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

    
    create_ml_flow_experiment(args.experiment_name)
    #need to add a logger

    #can add loss
    
    train(opt= args, env= env,model_path= model_path,device =device)
    
        
        




    


    
if __name__ == '__main__':
    main()
