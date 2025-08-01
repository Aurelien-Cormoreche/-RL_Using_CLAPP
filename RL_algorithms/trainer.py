import tqdm
import mlflow

from .trainer_utils import save_models_

from .actor_critic.train import actor_critic_train, actor_critic_metrics, actor_critic_log_params, actor_critic_modules, actor_critic_init
from .PPO.train import ppo_log_params, ppo_modules, ppo_collector, ppo_updator, ppo_metrics, ppo_init
from .Reinforce_baseline.train import reinforce_baseline_collector, reinforce_baseline_updator, reinforce_baseline_metrics, reinforce_baseline_modules, reinforce_baseline_init, reinforce_baseline_log_params

class Trainer:
    def __init__(self, opt, envs,  encoder, feature_dim, action_dim):
       
        self.opt = opt
        self.envs = envs
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.algorithm = opt.algorithm
        if self.algorithm == 'actor_critic' or self.algorithm == 'actor_critic_e':
            self.training_func = actor_critic_train
            self.call_func = self.__train_online
            self.metrics_func = actor_critic_metrics
            self.modules_func = actor_critic_modules
            self.ini_variables_func = actor_critic_init
            self.log_params_func = actor_critic_log_params
        elif self.algorithm == 'ppo':
            self.collector = ppo_collector
            self.updator = ppo_updator
            self.call_func = self.__train_offline
            self.metrics_func = ppo_metrics
            self.modules_func = ppo_modules
            self.ini_variables_func = ppo_init
            self.log_params_func = ppo_log_params
        elif self.algorithm == 'reinforce_baseline':
            self.collector = reinforce_baseline_collector
            self.updator = reinforce_baseline_updator
            self.call_func = self.__train_offline
            self.metrics_func = reinforce_baseline_metrics
            self.modules_func = reinforce_baseline_modules
            self.ini_variables_func = reinforce_baseline_init
            self.log_params_func = reinforce_baseline_log_params
        else:
            raise Exception('algorithm not found')
    
    def train(self):
        if self.opt.track_run:
            self.__log_params()
        self.models_dict = {}
        self.variables = self.__initialize_variables()
        self.modules = self.__create_modules()

        for self.epoch in tqdm.tqdm(range(self.opt.num_epochs)):
            self.variables = self.call_func()
            if self.opt.track_run:
                self.__log_metrics()
        if self.epoch % self.opt.checkpoint_interval == 0:
            agent = self.modules[0]
            icm = self.modules[1]
            save_models_(self.opt, self.models_dict, agent, icm)

    def __train_online(self):
       return self.training_func(self.opt, self.envs, self.modules, self.variables, self.epoch)

    def __train_offline(self):
        collected = self.collector(self.opt, self.envs, self.modules, self.variables, self.epoch)
        return self.updator(self.opt, self.modules, self.variables, collected)

    def __log_params(self):
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
        self.log_params_func(self.opt)
    
    def __create_modules(self):
        return self.modules_func(self.opt, self.variables, self.encoder, self.models_dict, self.envs)

    def __log_metrics(self):
        self.metrics_func(self.opt, self.epoch, self.variables)
    
    def __initialize_variables(self):
        return self.ini_variables_func(self.opt, self.feature_dim, self.action_dim, self.envs)


    
        