class Trainer:
    def __init__(self, opt, encoder, feature_dim, action_dim):
        
        online_algos = ['actor_critic', 'actor_critic_e']
        offline_algos = ['ppo', 'reinforce_baseline']
        
        self.opt = opt
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.algorithm = opt.algorithm
        if self.algorithm in online_algos:
            self.online = True
        if self.algorithm in offline_algos:
            self.online = False
        else:
            raise Exception('algorithm not found')
    
    def train(self):
        return

    def __train_online(self):
        return

    def __train_offline(self):
        return

    def __log_params(self):
        return
    
    def __create_modules(self):
        return

    def __log_parameters(self):
        return

    def __log_metrics(self):
        return


    
        