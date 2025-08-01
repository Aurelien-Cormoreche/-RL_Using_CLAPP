from utils.utils import create_envs, parsing, select_device
from utils.load_standalone_model import load_model
import math
import numpy as np
import torch 
import time
import tqdm._tqdm as tqdm
import os
class Specifications():
    def __init__(self):
        self.num_envs = 8
        self.num_turns = 4

        self.width = 2.74
        self.height = 2.74
        self.step = 2
        self.num_points = 1000
        self.size_features = 1024
        self.size_labels = 1


def teleport_agent(pos, dir, agent):
   agent.pos = pos.cpu()
   agent.dir = dir.cpu()

if __name__ == '__main__':

    args = parsing()
    device = select_device(args)
    specs = Specifications()
    assert args.greyscale
    envs = create_envs(args, specs.num_envs)
    encoder = load_model(os.path.abspath('trained_models')).to(device).eval().requires_grad_(False)

    deviations = torch.tensor(
        [[specs.width * x + 0.15, 0, -1, -math.pi / specs.num_turns] for x in range(3)]  \
        + [[8.37, 0, z * specs.height - 6.48,  -math.pi / specs.num_turns ] for z in range(5)],
    requires_grad= False, device= device)
    
    envs.reset()
    features_dataset = torch.empty((specs.num_points * specs.num_turns * specs.num_envs, specs.size_features), dtype= torch.float32, device= device)
    labels_dataset = torch.empty((specs.num_points * specs.num_turns * specs.num_envs, specs.size_labels ), dtype= torch.float32, device= device)
    for i in tqdm.tqdm(range(specs.num_points)):
        val = torch.rand((specs.num_envs, 4), requires_grad=False, device= device)  
        val *= torch.tensor([specs.step, 0, specs.step, math.pi * 2/ specs.num_turns], requires_grad=False, device= device)
        val += deviations
        idx = torch.arange(specs.num_envs, device= device) * specs.num_turns
        
        for t, agent in enumerate(envs.env.get_attr('agent')):
            teleport_agent(val[t, :-1], val[t, -1], agent)
    
        for j in range(specs.num_turns):
            obs = envs.step(np.zeros((specs.num_envs)))[0]
            val[:, -1] +=  math.pi * 2/ 4
            pos = i * specs.num_envs * specs.num_turns + j * specs.num_turns
            features = encoder(torch.tensor(np.expand_dims(obs, axis = 1), dtype= torch.float32, device= device))
            features_dataset[pos: pos + specs.num_envs] = features
            labels_dataset[pos: pos + specs.num_envs] = idx.unsqueeze(dim= -1)
            idx += 1

    features_dataset = features_dataset.cpu()
    labels_dataset = labels_dataset.cpu()
    torch.save(features_dataset,'dataset/T_maze_CLAPP_one_hot/features.pt')
    torch.save(labels_dataset,'dataset/T_maze_CLAPP_one_hot/labels.pt')
          
            
        




    

    

    

    



