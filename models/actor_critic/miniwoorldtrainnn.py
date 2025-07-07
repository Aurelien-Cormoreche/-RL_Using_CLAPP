import gymnasium as gym
from act_1layer_alg import ActCrit1Layer
import miniworld
import torch
import torch.nn.functional as F
import utils.load_standalone_model 
import os
from envs.T_maze.custom_T_Maze_V0 import myTmaze


if __name__ == "__main__":
   
    clapp = os.path.abspath("trained_models")
    
    maze = ()
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='custom_tmaze:MyTMaze',
    )
    env = gym.make("MyTMaze-v0", render_mode="human")
    model = ActCrit1Layer(env,clapp_model_path=clapp)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_episodes = 1800

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        model.clear_episode_data()
        optimizer.zero_grad()  # Clear gradients at the start of each episode

        while not done:
            action = model.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            model.store_reward(reward)
            old_state = state 
            state = next_state
            total_reward += reward
            done = done or truncated 
            # In newer gym versions, truncated can also end an episode 
         # Episode finished, now perform the training update
        loss = model.calculate_losses_and_update(optimizer)
        
        

        print(f"Episode {episode+1}, Total Reward: {total_reward}, Loss: {loss:.4f}")
        env.render()

    env.close()