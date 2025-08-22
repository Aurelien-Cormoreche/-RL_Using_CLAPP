import torch
import mlflow
from ..trainer_utils import get_features_from_state, defineScheduler
from ..agents import A_Agent
from utils.utils_torch import TorchDeque, CustomAdamEligibility

def reinforce_baseline_collector(opt, envs, modules, variables, epoch):
    """Collect experience using REINFORCE with baseline.

    Args:
        opt: Configuration object
        envs: Vectorized environments
        modules: Tuple of (agent, icm, optimizer, schedulers)
        variables: Tuple of training variables
        epoch: Current epoch number

    Returns:
        Updated training variables including advantage estimate
    """
    agent, icm, optimizer, schedulers = modules
    feature_dim, action_dim, moving_avg, *_ = variables
    baseline_schedule = schedulers[2]

    # Reset environment and initialize memory
    state, _ = envs.reset(seed=opt.seed + epoch * opt.seed)
    features = get_features_from_state(opt, state, agent, opt.device)
    memory = TorchDeque(maxlen=opt.nb_stacked_frames, num_features=feature_dim,
                       device=opt.device, dtype=torch.float32)
    memory.fill(features)

    # Run episode
    done = False
    total_reward = 0
    length_episode = 0
    optimizer.reset_z()

    while not done:
        # Get action and take step
        action, logprob, _ = agent.get_action_and_log_prob_dist_from_features(
            memory.get_all_content_as_tensor())

        for _ in range(opt.frame_skip):
            n_state, reward, terminated, truncated, _ = envs.step([action.detach().item()])
            length_episode += 1
            if terminated or truncated:
                break

        # Process rewards and states
        reward = reward[0]
        terminated, truncated = terminated[0], truncated[0]
        features = get_features_from_state(opt, n_state, agent, opt.device)
        memory.push(features)
        total_reward += reward
        done = terminated or truncated

        # Update optimizer
        optimizer.zero_grad()
        logprob.backward()
        with torch.no_grad():
            optimizer.accumulate()

        if opt.render:
            envs.render()

    # Calculate advantage
    moving_avg = moving_avg.lerp(
        torch.tensor(total_reward, device=opt.device, dtype=torch.float32),
        torch.tensor(baseline_schedule.get_lr(), device=opt.device, dtype=torch.float32))
    advantage = total_reward - moving_avg

    # Step schedulers
    for scheduler in schedulers:
        scheduler.step_forward()

    return feature_dim, action_dim, moving_avg, length_episode, total_reward, 0, 0, advantage

def reinforce_baseline_updator(opt, modules, variables, collected):
    """Update policy using collected experience.

    Args:
        opt: Configuration object
        modules: Tuple of training modules
        variables: Current training variables
        collected: Collected experience data

    Returns:
        Updated training variables
    """
    optimizer = modules[2]
    with torch.no_grad():
        optimizer.step(collected[-1], None)  # Update with advantage
    return *collected[:-1], 0, 0  # Return all except advantage

def reinforce_baseline_modules(opt, variables, encoder, models_dict, envs):
    """Initialize REINFORCE with baseline modules.

    Args:
        opt: Configuration object
        variables: Current training variables
        encoder: Feature encoder
        models_dict: Dictionary of models
        envs: Vectorized environments

    Returns:
        Tuple of (agent, icm, optimizer, schedulers)
    """
    feature_dim, action_dim, *_ = variables

    # Initialize agent and schedulers
    agent = A_Agent(feature_dim, action_dim, None, encoder, opt.normalize_features).to(opt.device)
    actor_lr_scheduler = defineScheduler(opt.schedule_type_actor, opt.actor_lr_i,
                                         opt.actor_lr_e, opt.num_epochs,
                                         opt.actor_lr_m, opt.actor_len_w)
    theta_lam_scheduler = defineScheduler(opt.schedule_type_theta_lam, opt.t_delay_theta_i,
                                         opt.t_delay_theta_e, opt.num_epochs,
                                         opt.theta_l_m, opt.theta_l_len_w)
    baseline_scheduler = defineScheduler(opt.schedule_type_baseline, opt.baseline_i,
                                          opt.baseline_e, opt.num_epochs)

    schedulers = (actor_lr_scheduler, theta_lam_scheduler, baseline_scheduler)
    optimizer = CustomAdamEligibility(agent.actor, opt.device, actor_lr_scheduler,
                                      theta_lam_scheduler, False, None, opt.gamma)

    return agent, None, optimizer, schedulers

def reinforce_baseline_metrics(opt, epoch, variables):
    """Log training metrics to MLflow.

    Args:
        opt: Configuration object
        epoch: Current epoch number
        variables: Current training variables
    """
    _, _, moving_avg, length_episode, total_reward, *_ = variables
    mlflow.log_metrics({
        'reward': total_reward,
        'length_episode': length_episode,
        'moving_avg': moving_avg.item()
    }, step=epoch)

def reinforce_baseline_init(opt, feature_dim, action_dim, envs):
    """Initialize training variables.

    Args:
        opt: Configuration object
        feature_dim: Dimension of feature space
        action_dim: Dimension of action space
        envs: Vectorized environments

    Returns:
        Initial training variables
    """
    return feature_dim, action_dim, torch.zeros((1), device=opt.device), 0, 0, 0, 0

def reinforce_baseline_log_params(opt):
    """Log hyperparameters to MLflow."""
    mlflow.log_params({
        # Actor learning rate parameters
        'schedule_type_actor': opt.schedule_type_actor,
        'actor_lr_i': opt.actor_lr_i,
        'actor_lr_e': opt.actor_lr_e,
        'actor_lr_m': opt.actor_lr_m,
        'actor_len_w': opt.actor_len_w,

        # Eligibility trace parameters
        'schedule_type_theta_lam': opt.schedule_type_theta_lam,
        't_delay_theta_i': opt.t_delay_theta_i,
        't_delay_theta_e': opt.t_delay_theta_e,
        'theta_l_m': opt.theta_l_m,
        'theta_l_len_w': opt.theta_l_len_w,

        # Baseline parameters
        'schedule_type_baseline': opt.schedule_type_baseline,
        'baseline_i': opt.baseline_i,
        'baseline_e': opt.baseline_e,
    })