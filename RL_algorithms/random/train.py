import random
import mlflow

def random_train(opt, envs, modules, variables, epoch):
    done = False
    num_actions = variables[0]
    step = 0
    rewards = 0
    envs.reset(seed = opt.seed + epoch * opt.seed)
    while not done:
        action = random.randint(0,num_actions)
        _ , reward, terminated, truncated, _ = envs.step([action])
        done = terminated or truncated
        rewards += reward
        step += 1
    return num_actions, step, rewards
    
        
def random_metrics(opt, epoch, variables):
    _, length_episode, total_reward = variables
    mlflow.log_metrics( 
                {
                    'reward': total_reward,
                    'length_episode': length_episode
                },
                step= epoch
            )
    
    
    
def random_log_params(opt):
    return
    
def random_modules(opt, variables, encoder, models_dict, envs):
    return
    
    
def random_init(opt, feature_dim, action_dim, envs):
    return  action_dim, 0, 0