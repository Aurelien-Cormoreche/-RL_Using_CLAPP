# Import necessary libraries for deep learning, argument parsing, environment management, and logging.
import torch
import math
import mlflow
import argparse
import os
import numpy as np
import gymnasium as gym
# Import custom modules for dimensionality reduction, model loading, and environment discretization.
# from sklearn.decomposition import PCA  # Commented out in favor of custom PCA implementation.
from utils.dimensionality_reduction import PCA
from mlflow import MlflowClient, MlflowException
from utils.load_standalone_model import load_model
from utils.tmaze_discretizer import TmazeDiscretizer

# Parse command-line arguments for environment and training configuration.
def parsing():
    parser = argparse.ArgumentParser()

    # Arguments for the environment configuration.
    parser.add_argument('--environment', default='envs.Rooms_4_maze.custom_Four_Maze_V0:FourRoomsMaze',
                        help='Path of the environment to use.')
    parser.add_argument('--greyscale', action='store_true',
                        help='Determine if we render the state in grayscale.')
    parser.add_argument('--render', action='store_true',
                        help='If set, the maze will be rendered during training.')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='The number of synchronous environments to spawn.')
    parser.add_argument('--visible_reward', action='store_true',
                        help='If the reward is a visible red box or not.')
    parser.add_argument('--max_episode_steps', default=1000, type=int,
                        help='Maximum number of steps per episode in the environment.')
    parser.add_argument('--no_images', action='store_true',
                        help='Whether to have the maze without any images.')
    parser.add_argument('--intermediate_rewards', action='store_true',
                        help='Whether to have intermediate rewards for exploration.')
    parser.add_argument('--no_reward', action='store_true',
                        help='Whether there should be rewards in the environment')
    # Arguments for the training configuration.
    parser.add_argument('--task', default='train',
                        help='Which task to perform (e.g., train, evaluate).')
    parser.add_argument('--algorithm', default='actor_critic',
                        help='Type of reinforcement learning algorithm to use.')
    parser.add_argument('--encoder', default="CLAPP",
                        help="Decide which encoder to use for feature extraction.")
    parser.add_argument('--keep_patches', action='store_true',
                        help='Keep the patches for the encoder.')
    parser.add_argument('--seed', default=1, type=int,
                        help='Manual seed for training reproducibility.')
    parser.add_argument('--log_models', action='store_true',
                        help='Whether to save the models during training.')
    parser.add_argument('--checkpoint_interval', default=1000, type=int,
                        help='Interval at which to save the model weights.')
    parser.add_argument('--save_name', default='saved_from_run.pt', type=str,
                        help='Name of the file where the model will be saved.')
    parser.add_argument('--two_layers', action='store_true',
                        help='If true, then actor and critic contain an activation layer.')

    # Hyperparameters for the training process.
    parser.add_argument('--num_epochs', default=80000, type=int,
                        help='Number of epochs for the training.')
    parser.add_argument('--gamma', default=0.995, type=float,
                        help='Discount factor (gamma) for training in the environment.')
    parser.add_argument('--nb_stacked_frames', default=1, type=int,
                        help='Number of stacked frames given as input.')
    parser.add_argument('--frame_skip', default=1, type=int,
                        help='Number of frames to skip during training.')
    parser.add_argument('--use_ICM', action='store_true',
                        help='Whether to use Intrinsic Curiosity Module (ICM) or not.')
    parser.add_argument('--encoder_layer', default='none',
                        help='Which encoder to use for feature extraction.')
    parser.add_argument('--encoder_lr', default=1e-4, type=float,
                        help='Learning rate for the models of the ICM.')
    parser.add_argument('--encoder_latent_dim', default=1024, type=int,
                        help='Latent dimension for ICM encoder.')
    parser.add_argument('--encoder_latent_dim_direction', default=16, type=int,
                        help='Latent dimension for direction encoder.')
    parser.add_argument('--encoder_latent_dim_time', default=128, type=int,
                        help='Latent dimension for time encoder.')
    parser.add_argument('--encoder_model_time', default='time_contrastive_encoder.pt', type=str,
                        help='Name of the time encoder model file.')
    parser.add_argument('--encoder_model_direction', default='direction_contrastive_encoder.pt', type=str,
                        help='Name of the direction encoder model file.')
    parser.add_argument('--encoder_output_mode', default='concatenate', type=str,
                        help='The type of the output of the encoder (e.g., concatenate, sum).')
    parser.add_argument('--alpha_intrinsic_reward', default=1e-1, type=float,
                        help='Coefficient for intrinsic reward in ICM.')
    parser.add_argument('--num_updates_encoder', default=1, type=int,
                        help='Number of updates for the ICM models per training step.')
    parser.add_argument('--PCA', action='store_true',
                        help='Use PCA for dimensionality reduction in ICM.')
    parser.add_argument('--lr_scheduler', action='store_true',
                        help='Add a learning rate scheduler.')
    parser.add_argument('--normalize_features', action='store_true',
                        help='Normalize the features from the encoder.')
    parser.add_argument('--target', action='store_true',
                        help='Whether to use a target network for stable training.')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Update rate for the target network.')
    parser.add_argument('--schedule_type_critic', default='constant',
                        help='Schedule type for the critic learning rate.')
    parser.add_argument('--critic_lr_i', type=float, default=1e-4,
                        help='Initial learning rate for the critic.')
    parser.add_argument('--critic_lr_e', type=float, default=9e-5,
                        help='End learning rate for the critic.')
    parser.add_argument('--critic_lr_m', type=float, default=9e-5,
                        help='Maximum critic learning rate (for warmup jobs).')
    parser.add_argument('--critic_len_w', type=int, default=10,
                        help='Warmup length for the critic learning rate scheduler.')
    parser.add_argument('--schedule_type_actor', default='constant',
                        help='Schedule type for the actor learning rate.')
    parser.add_argument('--actor_lr_i', type=float, default=9e-5,
                        help='Initial learning rate for the actor.')
    parser.add_argument('--actor_lr_e', type=float, default=1e-4,
                        help='End learning rate for the actor.')
    parser.add_argument('--actor_lr_m', type=float, default=1e-4,
                        help='Maximum actor learning rate (for warmup jobs).')
    parser.add_argument('--actor_len_w', type=int, default=100,
                        help='Warmup length for the actor learning rate scheduler.')
    parser.add_argument('--schedule_type_theta_lam', default='constant',
                        help='Schedule type for the actor eligibility trace delay.')
    parser.add_argument('--t_delay_theta_i', type=float, default=0.9,
                        help='Initial delay for actor in case of eligibility trace.')
    parser.add_argument('--t_delay_theta_e', type=float, default=0.9,
                        help='End delay for actor in case of eligibility trace.')
    parser.add_argument('--theta_l_m', type=float, default=0.9,
                        help='Maximum actor eligibility trace delay (for warmup jobs).')
    parser.add_argument('--theta_l_len_w', type=int, default=10,
                        help='Warmup length for actor eligibility trace delay.')
    parser.add_argument('--schedule_type_w_lam', default='constant',
                        help='Schedule type for the critic eligibility trace delay.')
    parser.add_argument('--t_delay_w_i', type=float, default=0.9,
                        help='Initial delay for critic in case of eligibility trace.')
    parser.add_argument('--t_delay_w_e', type=float, default=0.9,
                        help='End delay for critic in case of eligibility trace.')
    parser.add_argument('--w_l_m', type=float, default=0.9,
                        help='Maximum critic eligibility trace delay (for warmup jobs).')
    parser.add_argument('--w_l_len_w', type=int, default=10,
                        help='Warmup length for critic eligibility trace delay.')
    parser.add_argument('--schedule_type_baseline', default='constant',
                        help='Schedule type for the baseline if we run REINFORCE with artificial baseline.')
    parser.add_argument('--baseline_i', type=float, default=0.00005,
                        help='Initial baseline value.')
    parser.add_argument('--baseline_e', type=float, default=0.00005,
                        help='End baseline value.')
    parser.add_argument('--schedule_type_epsilon', default='constant',
                        help='Schedule type for epsilon if we use epsilon-greedy exploration.')
    parser.add_argument('--epsilon_i', type=float, default=0.1,
                        help='Initial epsilon value.')
    parser.add_argument('--epsilon_e', type=float, default=0.0,
                        help='End epsilon value.')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='Alpha for updating the Q values.')
    parser.add_argument('--threshold_pqueue', default=0.05, type=float,
                        help='Threshold for adding state-action pairs in prioritized sweeping.')
    parser.add_argument('--repeat_updates_p_sweep', default=5, type=int,
                        help='Number of updates in prioritized sweeping.')
    parser.add_argument('--entropy', action='store_true',
                        help='Add an entropy component to the loss for exploration.')
    parser.add_argument('--schedule_type_entropy', default='constant',
                        help='Schedule type for the entropy coefficient.')
    parser.add_argument('--coeff_entropy_i', type=float, default=0.0005,
                        help='Initial coefficient of the entropy in the PPO loss.')
    parser.add_argument('--coeff_entropy_e', type=float, default=0.005,
                        help='End coefficient of the entropy in the PPO loss.')
    parser.add_argument('--coeff_entropy_m', type=float, default=0.005,
                        help='Maximum coefficient of the entropy (for warmup schedule).')
    parser.add_argument('--coeff_entropy_len_w', type=int, default=2000,
                        help='Warmup length for the entropy coefficient schedule.')
    parser.add_argument('--len_rollout', default=1024, type=int,
                        help='Length of the continuous rollout.')
    parser.add_argument('--num_updates', default=8, type=int,
                        help='Number of optimization steps per rollout.')
    parser.add_argument('--minibatch_size', default=256, type=int,
                        help='Define minibatch size for offline learning.')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Learning rate if we need only one learning rate for the algorithm.')
    parser.add_argument('--lambda_gae', default=0.97, type=float,
                        help='Lambda used when calculating the Generalized Advantage Estimation (GAE).')
    parser.add_argument('--not_normalize_advantages', action='store_false',
                        help='Normalize the advantages of each minibatch.')
    parser.add_argument('--critic_eps', default=0.25, type=float,
                        help='Epsilon for clipping the critic updates in PPO.')
    parser.add_argument('--actor_eps', default=0.25, type=float,
                        help='Epsilon for clipping the actor updates in PPO.')
    parser.add_argument('--coeff_critic', default=0.5, type=float,
                        help='Coefficient of the critic in the PPO general loss.')
    parser.add_argument('--coeff_entropy', default=0.0005, type=float,
                        help='Coefficient of the entropy in the PPO general loss.')
    parser.add_argument('--grad_clipping', action='store_true',
                        help='Whether to clip the gradients during training.')

    # MLflow parameters for experiment tracking.
    parser.add_argument('--track_run', action='store_true',
                        help='Track the training run with MLflow.')
    parser.add_argument('--experiment_name', default='actor_critic_tMaze_default',
                        help='Name of the experiment on MLflow.')
    parser.add_argument('--run_name', default='default_run',
                        help='Name of the run on MLflow.')
    parser.add_argument('--experiment', action='store_true',
                        help='Run a full-scale MLflow experiment.')

    return parser.parse_args()

# Create vectorized environments for training.
def create_envs(args, num_envs, reward=True):
    # Register the custom environment.
    gym.envs.register(
        id='MyMaze',
        entry_point=args.environment
    )

    # Create vectorized environments.
    envs = gym.make_vec("MyMaze", num_envs=num_envs,
                        max_episode_steps=args.max_episode_steps,
                        render_mode='human' if args.render else None,
                        visible_reward=args.visible_reward,
                        reward=reward,
                        remove_images=args.no_images,
                        intermediate_rewards=args.intermediate_rewards)

    # Apply grayscale wrapper if specified.
    if args.greyscale:
        envs = gym.wrappers.vector.GrayscaleObservation(envs)

    return envs

# Launch an experiment with specified configurations and seeds.
def launch_experiment(opt, run_dicts, seeds, experiment_name, device, models_dict):
    from main import train  # Import the training function.

    # Create or set the MLflow experiment.
    create_ml_flow_experiment(experiment_name)
    model_path = os.path.abspath('trained_models')

    # Loop over seeds for reproducibility.
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'mps':
            torch.mps.manual_seed(seed)
        elif device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        else:
            print('Not possible to assign seed for this device.')

        # Loop over run configurations.
        for run_dict in run_dicts:
            for key in run_dict:
                setattr(opt, key, run_dict[key])

            # Create environments and start training.
            env = create_envs(opt, opt.num_envs)
            train(opt, env, model_path, device, models_dict)
            mlflow.end_run()

# Save trained models to a specified path.
def save_models(opt, models_dict):
    torch.save(models_dict, f"{os.environ['SAVED_MODELS_S2025']}/{opt.save_name}")

# Create or set an MLflow experiment for tracking runs.
def create_ml_flow_experiment(experiment_name, uri=f"file:{os.environ['ML_RUNS_S2025']}"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)

# Collect and store features from all positions in the environment using the specified encoder.
def collect_and_store_features(args, filename, encoder, env):
    disc = TmazeDiscretizer(env, encoder)
    features = disc.extract_features_from_all_positions()
    np.save(filename, features)
    return features

# Create a PCA model for dimensionality reduction.
def createPCA(args, filename, env, encoder, n_components, n_elements=-1):
    # Load or collect features.
    if os.path.exists(filename):
        if filename.endswith('pt'):
            features = torch.load(filename, map_location=args.device)
        else:
            features = torch.from_numpy(np.load(filename)).to(args.device)
    else:
        features = collect_and_store_features(args, filename, encoder, env)

    # Truncate features if necessary.
    features = features[:n_elements, :]

    # Fit PCA to the features.
    pca = PCA(size_input=features.shape[0], num_components=n_components)
    pca.fit(features)
    return pca

# Select the appropriate device (CPU, CUDA, or MPS) for training.
def select_device(args):
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
        print('CPU device selected; no seed set for GPU/MPS.')

    args.device = device
    return device