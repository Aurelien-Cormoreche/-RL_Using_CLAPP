import os

#from RL_algorithms.actor_critic.train import train_actor_critic
from RL_algorithms.PPO.train import train_PPO
from RL_algorithms.trainer import Trainer
from utils.load_standalone_model import load_model
from utils.utils import save_models, create_ml_flow_experiment, parsing, create_envs, launch_experiment, createPCA

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
import mlflow

def train(opt, envs, model_path, device, models_dict):
    
    gamma = opt.gamma

    if opt.encoder == 'CLAPP':
        encoder = load_model(model_path= model_path).eval()
        feature_dim = 1024
        if opt.keep_patches:
            feature_dim = 15 * 1024
    elif opt.encoder == 'resnet':    
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        encoder.fc = torch.nn.Identity()
        assert not opt.greyscale
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

    trainer = Trainer(opt, envs, encoder, feature_dim, action_dim)
    trainer.train()

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
    args.device = device


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
