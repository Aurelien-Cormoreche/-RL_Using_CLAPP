# Import necessary libraries for deep learning and custom utilities.
import torch
from RL_algorithms.trainer_utils import get_features_from_state  # Custom utility to extract features from environment state.
from utils.utils_torch import TorchDeque  # Custom circular buffer implementation for PyTorch tensors.

# Visualize the policy of a trained agent in the given environment.
# This function runs the agent's policy for a specified number of epochs and renders the environment.
# Parameters:
#   - opt: Configuration object containing hyperparameters and settings.
#   - envs: Vectorized environment(s) where the agent will act.
#   - agent: The trained agent whose policy will be visualized.
#   - num_epochs: Number of episodes/epochs to visualize.
def visualize_policy(opt, envs, agent, num_epochs):
    # Loop over the specified number of epochs.
    for epoch in range(num_epochs):
        # Reset the environment and get the initial state.
        state, _ = envs.reset()
        # Extract features from the initial state using the agent's encoder.
        features = get_features_from_state(opt, state, agent, opt.device)
        # Initialize a circular buffer (deque) to store stacked frames/features.
        memory = TorchDeque(
            maxlen=opt.nb_stacked_frames,  # Maximum number of stacked frames.
            num_features=1024,  # Number of features per frame.
            device=opt.device,  # Device (CPU/GPU) where tensors are stored.
            dtype=torch.float32  # Data type of the tensors.
        )
        # Fill the deque with the initial features to ensure it's full before starting.
        memory.fill(features)

        # Initialize the done flag to control the episode loop.
        done = False
        # Initialize episode length counter.
        length_episode = 0
        # Loop until the episode is done (terminated or truncated).
        while not done:
            # Get the action and log probabilities from the agent's policy using the current features.
            action, _, _ = agent.get_action_and_log_prob_dist_from_features(
                memory.get_all_content_as_tensor()  # Get all features in the deque as a single tensor.
            )
            # Execute the action in the environment for the specified number of frame skips.
            for _ in range(opt.frame_skip):
                # Step the environment with the chosen action.
                n_state, _, terminated, truncated, _ = envs.step([action.detach().item()])
                length_episode += 1  # Increment the episode length counter.
                # Break if the episode is terminated or truncated.
                if terminated or truncated:
                    break

            # Extract the first element of the terminated and truncated arrays (for vectorized environments).
            terminated = terminated[0]
            truncated = truncated[0]

            # Extract features from the new state.
            features = get_features_from_state(opt, n_state, agent, opt.device)
            # Push the new features into the deque.
            memory.push(features)

            # Update the done flag based on termination or truncation.
            done = terminated or truncated

            # Render the environment if specified in the options.
            if opt.render:
                envs.render()