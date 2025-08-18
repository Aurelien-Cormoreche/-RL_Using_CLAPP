import torch
from RL_algorithms.trainer_utils import get_features_from_state
from utils.utils_torch import TorchDeque

def visualize_policy(opt, envs, agent, num_epochs):
    for epoch in range(num_epochs):
        state, _ = envs.reset()
        features = get_features_from_state(opt, state, agent, opt.device)
        memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= 1024, device= opt.device, dtype= torch.float32)
        memory.fill(features)
            
        done = False
        while not done:

            action, _, _ = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())

            for _ in range(opt.frame_skip):
                n_state, _, terminated, truncated, _ = envs.step([action.detach().item()])
                length_episode += 1
                if terminated or truncated:
                    break
                        
            terminated = terminated[0]
            truncated = truncated[0]
        
            features = get_features_from_state(opt, n_state, agent, opt.device)
            memory.push(features)
            
            done= terminated or truncated 
            
            if opt.render:
                envs.render()
        
        
    