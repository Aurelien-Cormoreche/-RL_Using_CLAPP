from ..trainer_utils import get_features_from_state, defineScheduler
from ..agents import Discrete_Model_Based_Agent
import heapq
import torch
import mlflow


def prioritized_sweeping_train(opt, envs, modules, variables, epoch):
    agent, espilon_schedulder = modules
    n_state, _ = envs.reset(seed = opt.seed + epoch * opt.seed)
    probability_state = get_features_from_state(opt, n_state, agent, opt.device)
    c_state = torch.distributions.Categorical(probs= probability_state).sample().item()      
    done = False
    num_states, num_actions, pqueue, _, _ = variables
    run_length = 0
    rewards = 0
    while not done:
        old_state = c_state
        action = agent.get_action_from_state(c_state)
        for _ in range(opt.frame_skip):
            n_state, reward, terminated, truncated, _ = envs.step([action])
            reward = reward[0]
            terminated = terminated[0]
            truncated = truncated[0]
            rewards += reward
            run_length += 1
            if terminated or truncated:
                break
       
        probability_state = get_features_from_state(opt, n_state, agent, opt.device)           
        c_state = torch.distributions.Categorical(probs= probability_state).sample().item()  
        if terminated or truncated:
            c_state = num_states
        agent.world_model.add(old_state, action, c_state, reward)
        p = abs(reward + opt.gamma * agent.max_val(c_state) - agent.val(old_state, action))
        if p > opt.threshold_pqueue:
            heapq.heappush(pqueue, (-p, (old_state, action)))
      
        for n in range(opt.repeat_updates_p_sweep):
            if len(pqueue) == 0:
                break
            (p, (s, a)) = heapq.heappop(pqueue)
           
            n_s, r = agent.world_model.predict(s,a)
            agent.update_q(s, a , n_s, r)
            for p_s, p_a in agent.world_model.leading_to(s):
                p  = abs(agent.world_model.predicted_reward(p_s, p_a) + opt.gamma  * agent.max_val(s) - agent.val(p_s, p_a))
                if p > opt.threshold_pqueue:
                    heapq.heappush(pqueue, (-p, (p_s, p_a)))
       
        done = terminated or truncated


        if opt.render:
            envs.render()

    print(run_length)
    espilon_schedulder.step_forward()
    return num_states, num_actions, pqueue, run_length, rewards



def prioritized_sweeping_modules(opt, variables, encoder, models_dict, envs):
    num_states, num_actions, pqueue, _, _  = variables
    espilon_schedulder = defineScheduler(opt.schedule_type_epsilon, opt.epsilon_i,  opt.epsilon_e, opt.num_epochs)
    agent = Discrete_Model_Based_Agent(num_states + 1, num_actions, encoder, espilon_schedulder, opt.alpha, opt.gamma)
    return agent, espilon_schedulder

def prioritized_sweeping_init(opt, feature_dim, action_dim, envs):
    return feature_dim, action_dim, [], None, None

def prioritized_sweeping_metrics(opt, epoch, variables):
    _, _, _, length_episode, total_reward = variables
    mlflow.log_metrics( 
                {
                    'reward': total_reward,
                    'length_episode': length_episode
                },
                step= epoch
            )
    
def prioritized_sweeping_log_params(opt):
    mlflow.log_params(
        {
            'alpha' : opt.alpha,
            'schedule_type_epsilon': opt.schedule_type_epsilon,
            'epsilon_i': opt.epsilon_lr_i,
            'epsilon_e': opt.epsilon_lr_e,
        }
    )
