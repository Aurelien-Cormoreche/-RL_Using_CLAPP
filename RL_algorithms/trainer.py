import tqdm
import mlflow

from .trainer_utils import save_models_

from .actor_critic.train import actor_critic_train, actor_critic_metrics, actor_critic_log_params, actor_critic_modules, actor_critic_init
from .PPO.train import ppo_log_params, ppo_modules, ppo_collector, ppo_updator, ppo_metrics, ppo_init
from .Reinforce_baseline.train import reinforce_baseline_collector, reinforce_baseline_updator, reinforce_baseline_metrics, reinforce_baseline_modules, reinforce_baseline_init, reinforce_baseline_log_params
from .prioritized_sweeping.train import prioritized_sweeping_init, prioritized_sweeping_log_params, prioritized_sweeping_metrics, prioritized_sweeping_modules, prioritized_sweeping_train
from .random.train import random_init, random_log_params, random_metrics, random_modules, random_train

class Trainer:
    """
    A unified training interface for multiple RL algorithms (online and offline).
    Supports: Actor-Critic, PPO, REINFORCE with baseline, Prioritized Sweeping, and Random policy.
    Handles:
    - Variable initialization
    - Module creation (models, encoders, ICM)
    - Logging with MLflow
    - Training loop
    """
    def __init__(self, opt, envs,  encoder, feature_dim, action_dim):
        """
        Initialize trainer with options, environment, and encoder.

        Parameters:
        - opt: config options containing hyperparameters and flags
        - envs: Vectorized environments for RL (gym style interactions)
        - encoder: Feature encoder (e.g., CLAPP, ResNet, raw pixels)
        - feature_dim: Dimension of encoder output
        - action_dim: Number of actions in environment
        """

        self.opt = opt
        self.envs = envs
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        # Select the algorithm and bind corresponding functions
        self.algorithm = opt.algorithm
        if self.algorithm == 'actor_critic' or self.algorithm == 'actor_critic_e':
            # Online Actor-Critic training
            self.training_func = actor_critic_train
            self.call_func = self.__train_online
            self.metrics_func = actor_critic_metrics
            self.modules_func = actor_critic_modules
            self.ini_variables_func = actor_critic_init
            self.log_params_func = actor_critic_log_params
        elif self.algorithm == 'ppo':
            # Offline PPO training
            self.collector = ppo_collector
            self.updator = ppo_updator
            self.call_func = self.__train_offline
            self.metrics_func = ppo_metrics
            self.modules_func = ppo_modules
            self.ini_variables_func = ppo_init
            self.log_params_func = ppo_log_params
        elif self.algorithm == 'reinforce_baseline':
            # Offline REINFORCE with baseline
            self.collector = reinforce_baseline_collector
            self.updator = reinforce_baseline_updator
            self.call_func = self.__train_offline
            self.metrics_func = reinforce_baseline_metrics
            self.modules_func = reinforce_baseline_modules
            self.ini_variables_func = reinforce_baseline_init
            self.log_params_func = reinforce_baseline_log_params
        elif self.algorithm == 'prioritized_sweeping':
            # Online Prioritized Sweeping
            self.training_func = prioritized_sweeping_train
            self.call_func = self.__train_online
            self.metrics_func = prioritized_sweeping_metrics
            self.modules_func = prioritized_sweeping_modules
            self.ini_variables_func = prioritized_sweeping_init
            self.log_params_func = prioritized_sweeping_log_params
        elif self.algorithm == 'random':
            # Random policy
            self.training_func = random_train
            self.call_func = self.__train_online
            self.metrics_func = random_metrics
            self.modules_func = random_modules
            self.ini_variables_func = random_init
            self.log_params_func = random_log_params
        else:
            raise Exception('algorithm not found')
    
    def train(self):
        """
        Main training loop. Handles:
        - Parameter logging
        - Variable initialization
        - Module creation
        - Iterative training across epochs
        - Periodic model checkpointing
        """

        #log the parameters to track the run
        if self.opt.track_run:
            self.__log_params()
        self.models_dict = {}
        #initialize the variables and modules needed for training
        self.variables = self.__initialize_variables()
        self.modules = self.__create_modules()
        
        # Epoch loop
        for self.epoch in tqdm.tqdm(range(self.opt.num_epochs)):
            # Call either online or offline training function
            self.variables = self.call_func()
            # log metrics to track the run
            if self.opt.track_run:
                self.__log_metrics()
            # log the models if needed
            if self.opt.log_models and self.epoch % self.opt.checkpoint_interval == 0:
                agent = self.modules[0]
                icm = None
                if self.opt.use_ICM:
                    icm = self.modules[1]
                save_models_(self.opt, self.models_dict, agent, icm)
    # ----- Internal helper functions -----
    def __train_online(self):
        """
        Execute one online training step (Actor-Critic, Prioritized Sweeping, Random)
        Update network after each step
        """
        return self.training_func(self.opt, self.envs, self.modules, self.variables, self.epoch)

    def __train_offline(self):
        """
        Execute one offline training step (PPO, REINFORCE with baseline):
        1. Collect trajectories
        2. Update networks
        """
        collected = self.collector(self.opt, self.envs, self.modules, self.variables, self.epoch)
        return self.updator(self.opt, self.modules, self.variables, collected)

    def __log_params(self):        
        """
        Log hyperparameters and options to MLflow
        """
        opt = self.opt
        if opt.track_run:
            mlflow.start_run(run_name= opt.run_name)
            mlflow.log_params(
                    {
                        'algorithm' : opt.algorithm,
                        'num_envs' : opt.num_envs,
                        'greyscale' : opt.greyscale,
                        'encoder': opt.encoder,
                        'num_epochs': opt.num_epochs,
                        'gamma': opt.gamma,
                        'keep_patches' : opt.keep_patches, 
                        'seed' : opt.seed,
                        'visible_reward' : opt.visible_reward,
                        'normalize_features' : opt.normalize_features             
                    }
            )
        # Call algorithm-specific logging function
        self.log_params_func(self.opt)
    
    def __create_modules(self):
        """
        Instantiate the RL modules (actor, critic, ICM, etc.) based on the algorithm
        """
        return self.modules_func(self.opt, self.variables, self.encoder, self.models_dict, self.envs)

    def __log_metrics(self):
        """
        Log metrics for the current epoch
        """
        self.metrics_func(self.opt, self.epoch, self.variables)
    
    def __initialize_variables(self):
        """
        Initialize buffers, states, and other variables for training
        """
        return self.ini_variables_func(self.opt, self.feature_dim, self.action_dim, self.envs)


    
        