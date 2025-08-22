# Random training and logging utilities for reinforcement learning experiments.
import random
import mlflow

def random_train(opt, envs, modules, variables, epoch):
    """Run a single random episode in the environment.

    Args:
        opt: Configuration object with training settings.
        envs: Vectorized environment(s).
        modules: Unused (for compatibility).
        variables: Tuple containing number of possible actions.
        epoch: Current epoch number (used for seeding).

    Returns:
        Tuple: (num_actions, steps_taken, total_reward)
    """
    done = False
    num_actions = variables[0]
    step, rewards = 0, 0
    envs.reset(seed=opt.seed + epoch * opt.seed)

    while not done:
        action = random.randint(0, num_actions)
        _, reward, terminated, truncated, _ = envs.step([action])
        done = terminated or truncated
        rewards += reward
        step += 1

    return num_actions, step, rewards

def random_metrics(opt, epoch, variables):
    """Log random training metrics to MLflow.

    Args:
        opt: Configuration object.
        epoch: Current epoch number.
        variables: Tuple containing (_, episode_length, total_reward).
    """
    _, length_episode, total_reward = variables
    mlflow.log_metrics(
        {'reward': total_reward, 'length_episode': length_episode},
        step=epoch
    )

def random_log_params(opt):
    """Placeholder for logging random training parameters."""
    return

def random_modules(opt, variables, encoder, models_dict, envs):
    """Placeholder for module initialization/updates in random training."""
    return

def random_init(opt, feature_dim, action_dim, envs):
    """Initialize variables for random training.

    Returns:
        Tuple: (action_dim, 0, 0) - action space size and placeholders
    """
    return action_dim, 0, 0