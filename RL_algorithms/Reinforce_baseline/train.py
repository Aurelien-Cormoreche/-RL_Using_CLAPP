import torch
import mlflow
from ..trainer_utils import get_features_from_state, defineScheduler
from ..agents import A_Agent
from utils.utils_torch import TorchDeque, CustomAdamEligibility

def reinforce_baseline_collector(opt, envs, modules, variables, epoch):
    agent, icm, optimizer, schedulders = modules
    feature_dim, action_dim, moving_avg, _ , _, _, _  = variables

    baseline_schedule = schedulders[2]
    
    state, _ = envs.reset(seed = opt.seed + epoch * opt.seed)
    features = get_features_from_state(opt, state, agent, opt.device)
    memory = TorchDeque(maxlen= opt.nb_stacked_frames, num_features= feature_dim, device= opt.device, dtype= torch.float32)
    memory.fill(features)
        
    done = False
    total_reward = 0
    length_episode = 0
    tot_loss_critic = 0
    tot_loss_actor = 0

    optimizer.reset_z()

    while not done:   
        action, logprob, dist = agent.get_action_and_log_prob_dist_from_features(memory.get_all_content_as_tensor())
   
        for _ in range(opt.frame_skip):
            n_state, reward, terminated, truncated, _ = envs.step([action.detach().item()])
            length_episode += 1
            if terminated or truncated:
                break
                       
        reward = reward[0]
        terminated = terminated[0]
        truncated = truncated[0]
            
        features = get_features_from_state(opt, n_state, agent, opt.device)
        memory.push(features)
        
        total_reward += reward
        done= terminated or truncated 
        
        optimizer.zero_grad()
        logprob.backward()
        with torch.no_grad():
            optimizer.accumulate()
        
        if opt.render:
            envs.render()
    moving_avg = moving_avg.lerp(torch.tensor(total_reward, device= opt.device, dtype= torch.float32), torch.tensor(baseline_schedule.get_lr(), device= opt.device, dtype= torch.float32))
    advantage = total_reward - moving_avg
    
    for scheduler in schedulders: scheduler.step_forward()

    return feature_dim, action_dim, moving_avg, length_episode, total_reward, tot_loss_actor, tot_loss_critic, advantage 


def reinforce_baseline_updator(opt, modules, variables, collected):
    feature_dim, action_dim, moving_avg, length_episode, total_reward, tot_loss_actor, tot_loss_critic, advantage = collected
    optimizer = modules[2]
    with torch.no_grad():
        optimizer.step(advantage, None)
    return feature_dim, action_dim, moving_avg, length_episode, total_reward, tot_loss_actor, tot_loss_critic

def reinforce_baseline_modules(opt, variables, encoder, models_dict, envs):
    feature_dim, action_dim, _, _, _, _, _ = variables
    agent = A_Agent(feature_dim, action_dim, None, encoder, opt.normalize_features).to(opt.device)
    actor_lr_scheduler = defineScheduler(opt.schedule_type_actor, opt.actor_lr_i, opt.actor_lr_e, opt.num_epochs, opt.actor_lr_m, opt.actor_len_w)
    theta_lam_scheduler =defineScheduler(opt.schedule_type_theta_lam, opt.t_delay_theta_i, opt.t_delay_theta_e, opt.num_epochs, opt.theta_l_m, opt.theta_l_len_w)
    baseline_scheduler = defineScheduler(opt.schedule_type_baseline, opt.baseline_i, opt.baseline_e, opt.num_epochs)
    schedulers = (actor_lr_scheduler, theta_lam_scheduler, baseline_scheduler)
    optimizer = CustomAdamEligibility(agent.actor, opt.device, actor_lr_scheduler, theta_lam_scheduler, False, None, opt.gamma)

    return agent, None, optimizer, schedulers

def reinforce_baseline_metrics(opt, epoch, variables):
    _, _, moving_avg, length_episode, total_reward, _, _ = variables
    mlflow.log_metrics( 
                {
                    'reward': total_reward,
                    'length_episode': length_episode,
                    'moving_avg' : moving_avg.item()
                },
                step= epoch
            )
def reinforce_baseline_init(opt, feature_dim, action_dim, envs):
    return  feature_dim, action_dim, torch.zeros((1), device= opt.device), 0, 0, 0, 0

def reinforce_baseline_log_params(opt):
    mlflow.log_params({
        # Actor learning rate scheduler
        'schedule_type_actor': opt.schedule_type_actor,
        'actor_lr_i': opt.actor_lr_i,
        'actor_lr_e': opt.actor_lr_e,
        'actor_lr_m': opt.actor_lr_m,
        'actor_len_w': opt.actor_len_w,
        # Actor eligibility trace scheduler
        'schedule_type_theta_lam': opt.schedule_type_theta_lam,
        't_delay_theta_i': opt.t_delay_theta_i,
        't_delay_theta_e': opt.t_delay_theta_e,
        'theta_l_m': opt.theta_l_m,
        'theta_l_len_w': opt.theta_l_len_w,
        # baseline schedulder
        'schedule_type_baseline': opt.schedule_type_baseline,
        'baseline_i': opt.baseline_i,
        'baseline_e': opt.baseline_e,
    })