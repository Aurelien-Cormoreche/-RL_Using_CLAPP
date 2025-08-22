# Import necessary libraries for mathematical operations, deep learning, and optimization.
import math
import torch
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
import torch.nn as nn
import random

# InfoNceLoss: Implements the InfoNCE (Noise-Contrastive Estimation) loss for contrastive learning.
# This loss is used to maximize the similarity between positive pairs and minimize similarity with negative pairs.
class InfoNceLoss():
    def __call__(self, real, positive, negatives):
        # Compute cosine similarity between the real sample and all positive/negative samples.
        similarities = torch.cosine_similarity(real, torch.cat((positive, negatives), dim=0)).unsqueeze(0)
        # Use CrossEntropyLoss to classify the positive sample as the first class.
        criterion = nn.CrossEntropyLoss()
        # The target is 0 because the positive sample is the first element in the concatenated tensor.
        return criterion(similarities, torch.tensor([0], device=similarities.device))

# TorchDeque: A circular buffer (double-ended queue) implemented using PyTorch tensors.
# This class is designed for efficient storage and sampling of data in reinforcement learning.
class TorchDeque():
    def __init__(self, maxlen, num_features, dtype, device):
        # maxlen: Maximum number of elements the deque can hold.
        # num_features: Number of features per element.
        # dtype: Data type of the elements.
        # device: Device (CPU/GPU) where the tensor is stored.
        self.maxlen = maxlen
        self.num_features = num_features
        self.dtype = dtype
        self.device = device
        # Initialize an empty tensor to store the elements.
        self.memory = torch.empty((maxlen, num_features), dtype=dtype, device=device)
        # Index to track the current position in the deque.
        self.index = 0
        # Current number of elements in the deque.
        self.size = 0
        # Index of the oldest element in the deque.
        self.start = 0

    # Fill the deque with a repeated tensor.
    def fill(self, data):
        self.memory = data.repeat(self.maxlen, 1)
        self.size = self.maxlen

    # Push a new element into the deque.
    # If the deque is full, the oldest element is overwritten.
    def push(self, data):
        if self.size == self.maxlen:
            # Move the start index forward if the deque is full.
            self.start = (self.start + 1) % self.maxlen
        else:
            # Increase the size if the deque is not full.
            self.size += 1
        # Store the old element before overwriting.
        old = self.memory[self.index]
        # Insert the new element.
        self.memory[self.index] = data
        # Move the index forward.
        self.index = (self.index + 1) % self.maxlen
        return old

    # Sample a specified number of elements randomly from the deque.
    def sample(self, num_samples):
        # Generate random indices for sampling.
        indices = torch.randperm(torch.tensor(min(num_samples, self.size)))[:num_samples]
        return self.memory[indices]

    # Return all elements in the deque as a flattened tensor.
    def get_all_content_as_tensor(self):
        # Roll the tensor to align the oldest element at the start.
        return torch.roll(self.memory, -self.start, dims=0).flatten()

    # Return the current size of the deque.
    def __sizeof__(self):
        return self.size

    # Reset the deque to its initial empty state.
    def reset(self):
        self.memory = torch.empty((self.maxlen, self.num_features), dtype=self.dtype, device=self.device)
        self.index = 0
        self.size = 0
        self.start = 0

# CascadeTime_Memory: A memory system with three cascading deques (recent, intermediate, old).
# This is used to store and sample experiences at different time scales.
class CascadeTime_Memory():
    def __init__(self, memory_sizes, num_features, device):
        # memory_sizes: List of sizes for the three deques.
        # num_features: Number of features per element.
        # device: Device (CPU/GPU) where the tensors are stored.
        self.memory_sizes = memory_sizes
        self.num_features = num_features
        self.device = device
        # Initialize the three deques.
        self.recent = TorchDeque(memory_sizes[0], num_features, torch.float32, device)
        self.intermediate = TorchDeque(memory_sizes[1], num_features, torch.float32, device)
        self.old = TorchDeque(memory_sizes[2], num_features, torch.float32, device)

    # Push a new element into the cascading memory.
    # The element moves from recent to intermediate to old as new elements are added.
    def push(self, data):
        x = self.recent.push(data)
        if x is not None:
            x = self.intermediate.push(x)
        if x is not None:
            x = self.old.push(x)

    # Sample positive examples from the recent deque.
    def sample_posititves(self, num_samples, direction):
        return self.recent.sample(num_samples)

    # Sample negative examples from the old deque.
    def sample_negatives(self, num_samples, direction):
        return self.old.sample(num_samples)

    # Check if the old deque is full.
    def full(self):
        return self.old.size == self.memory_sizes[2]

    # Reset all deques to their initial empty state.
    def reset(self):
        self.recent = TorchDeque(self.memory_sizes[0], self.num_features, torch.float32, self.device)
        self.intermediate = TorchDeque(self.memory_sizes[1], self.num_features, torch.float32, self.device)
        self.old = TorchDeque(self.memory_sizes[2], self.num_features, torch.float32, self.device)

    # Check if the old deque has enough elements for sampling.
    def can_sample(self, num_samples):
        return self.old.size >= num_samples

# Cascade_Direction_Memory: A memory system with separate deques for each direction.
# This is used to store and sample experiences based on direction.
class Cascade_Direction_Memory():
    def __init__(self, memory_sizes, num_features, device, eps):
        # memory_sizes: Size of each direction deque.
        # num_features: Number of features per element.
        # device: Device (CPU/GPU) where the tensors are stored.
        # eps: Probability of randomly pushing an element into a full deque.
        self.memory_sizes = memory_sizes
        self.num_features = num_features
        self.device = device
        self.eps = eps
        # Initialize a deque for each direction (8 directions).
        self.direction_0 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_1 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_2 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_3 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_4 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_5 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_6 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        self.direction_7 = TorchDeque(memory_sizes, num_features, torch.float32, device)
        # List of all direction deques for easy iteration.
        self.buffers = [self.direction_0, self.direction_1, self.direction_2, self.direction_3,
                        self.direction_4, self.direction_5, self.direction_6, self.direction_7]

    # Reset all direction deques to their initial empty state.
    def reset(self):
        for d in self.buffers:
            d.reset()

    # Push an element into the deque corresponding to the given direction.
    # If the deque is full, the element is only pushed with probability eps.
    def push(self, data, direction):
        if self.buffers[direction].size < self.memory_sizes or random.random() < self.eps:
            self.buffers[direction].push(data)

    # Sample elements from the deque corresponding to the given direction.
    def sample_direction(self, direction, num_samples):
        if self.buffers[int(direction)].size < num_samples:
            raise Exception('not enough elements in direction')
        return self.buffers[int(direction)].sample(num_samples)

    # Sample positive examples from the deque corresponding to the given direction.
    def sample_posititves(self, num_samples, direction):
        return self.sample_direction(direction, num_samples)

    # Check if all direction deques have enough elements for sampling.
    def can_sample(self, num_samples):
        for b in self.buffers:
            if b.size < num_samples:
                return False
        return True

    # Sample negative examples from all directions except the given direction and its neighbors.
    def sample_negatives(self, num_samples, direction):
        # Directions to exclude: the given direction and its immediate neighbors.
        non_negatives_directions = torch.tensor([direction - 1 % 8, direction, direction + 1 % 8], dtype=torch.float32, device=self.device)
        possible_directions = torch.arange(0, 8, device=self.device, dtype=torch.float32)
        # Create a mask to exclude the non-negative directions.
        mask = ~torch.isin(possible_directions, non_negatives_directions)
        directions_to_sample_from = possible_directions[mask]
        # Randomly select directions to sample from.
        to_sample = torch.ones((8 - 3)).multinomial(num_samples, replacement=True)
        toret = torch.empty((num_samples, self.num_features), dtype=torch.float32, device=self.device)
        for i in range(num_samples):
            toret[i] = self.sample_direction(directions_to_sample_from[to_sample[i]], 1)
        return toret

# CosineAnnealingWarmupLr: A learning rate scheduler that combines warmup and cosine annealing.
# This is used to gradually increase the learning rate during warmup and then decrease it cosinely.
class CosineAnnealingWarmupLr(SequentialLR):
    def __init__(self, optimizer, warmup_steps, total_steps, start_factor=1e-3, last_epoch=-1, eta_min=1e-6):
        # warmup: Linear learning rate warmup.
        self.warmup = LinearLR(optimizer, start_factor, 1, warmup_steps)
        # cosineAnnealing: Cosine annealing of the learning rate after warmup.
        self.cosineAnnealing = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
        # Combine the two schedulers sequentially.
        super().__init__(optimizer, [self.warmup, self.cosineAnnealing], [warmup_steps], last_epoch)

# CustomLrScheduler: Base class for custom learning rate schedulers.
class CustomLrScheduler():
    def __init__(self):
        # step: Current step count.
        self.step = 0

    # Get the current learning rate.
    def get_lr(self):
        raise NotImplementedError

    # Increment the step count.
    def step_forward(self):
        self.step += 1

# CustomLrSchedulerCosineAnnealing: Cosine annealing learning rate scheduler.
class CustomLrSchedulerCosineAnnealing(CustomLrScheduler):
    def __init__(self, base_lr, T_max, eta_min=0):
        super().__init__()
        # T_max: Maximum number of iterations.
        # eta_min: Minimum learning rate.
        # base_lr: Base learning rate.
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = base_lr

    # Compute the current learning rate using cosine annealing.
    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step / self.T_max)) / 2

# CustomLrSchedulerLinear: Linear learning rate scheduler.
class CustomLrSchedulerLinear(CustomLrScheduler):
    def __init__(self, initial_lr, end_lr, T_max):
        super().__init__()
        # initial_lr: Initial learning rate.
        # end_lr: Final learning rate.
        # T_max: Maximum number of iterations.
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.T_max = T_max

    # Compute the current learning rate using linear interpolation.
    def get_lr(self):
        return self.initial_lr + (self.step) / (self.T_max) * (self.end_lr - self.initial_lr)

# CustomComposeSchedulers: Combines multiple learning rate schedulers.
class CustomComposeSchedulers(CustomLrScheduler):
    def __init__(self, schedulers, milestones):
        super().__init__()
        # schedulers: List of learning rate schedulers.
        # milestones: List of step counts at which to switch schedulers.
        self.current_idx = 0
        self.schedulers = schedulers
        self.milestones = milestones

    # Get the current learning rate from the active scheduler.
    def get_lr(self):
        return self.schedulers[self.current_idx].get_lr()

    # Increment the step count and switch schedulers if a milestone is reached.
    def step_forward(self):
        super().step_forward()
        if self.step >= self.milestones[self.current_idx + 1]:
            self.current_idx += 1

# CustomWarmupCosineAnnealing: Combines warmup and cosine annealing learning rate schedulers.
class CustomWarmupCosineAnnealing(CustomComposeSchedulers):
    def __init__(self, initial_lr, max_lr, len_warmup, tot_len, eta_min):
        # warmupLr: Linear warmup scheduler.
        cosineAnnealing = CustomLrSchedulerCosineAnnealing(max_lr, tot_len - len_warmup, eta_min)
        warmupLr = CustomLrSchedulerLinear(initial_lr, max_lr, len_warmup)
        # Combine the two schedulers.
        super().__init__([warmupLr, cosineAnnealing], [0, len_warmup])

# CustomAdamDuoEligibility: Custom Adam optimizer with eligibility traces for both actor and critic.
class CustomAdamDuoEligibility():
    def __init__(self, actor, critic, device, lr_w_schedule, lr_theta_schedule, beta1_w_schedule, beta1_theta_schedule, entropy, entropy_scheduler, gamma, use_second_order=False, beta2=0.999):
        # Initialize Adam optimizers for actor and critic with eligibility traces.
        self.adam_theta = CustomAdamEligibility(actor, device, lr_theta_schedule, beta1_theta_schedule, entropy, entropy_scheduler, gamma, use_second_order, beta2)
        self.adam_w = CustomAdamEligibility(critic, device, lr_w_schedule, beta1_w_schedule, False, None, gamma, use_second_order, beta2)

    # Reset the eligibility traces for both actor and critic.
    def reset_zw_ztheta(self):
        self.adam_theta.reset_z()
        self.adam_w.reset_z()

    # Accumulate gradients and perform an optimization step for both actor and critic.
    def accumulate_and_step(self, advantage, entropy):
        self.adam_theta.accumulate()
        self.adam_theta.step(advantage, entropy)
        self.adam_w.accumulate()
        self.adam_w.step(advantage, entropy)

    # Perform an optimization step for both actor and critic.
    def step(self, advantage, entropy):
        self.adam_theta.step(advantage, entropy)
        self.adam_w.step(advantage, entropy)

    # Zero the gradients for both actor and critic.
    def zero_grad(self):
        self.adam_theta.zero_grad()
        self.adam_w.zero_grad()

# CustomAdamEligibility: Custom Adam optimizer with eligibility traces.
class CustomAdamEligibility():
    def __init__(self, model, device, lr_schedule, beta1_schedule, entropy, entropy_scheduler, gamma, use_second_order=False, beta2=0.999):
        # model: The model to optimize.
        # device: Device (CPU/GPU) where the tensors are stored.
        # lr_schedule: Learning rate scheduler.
        # beta1_schedule: Schedule for the first moment coefficient.
        # entropy: Whether to use entropy regularization.
        # entropy_scheduler: Scheduler for the entropy coefficient.
        # gamma: Discount factor for eligibility traces.
        # use_second_order: Whether to use second-order moments (like Adam).
        # beta2: Coefficient for the second moment (if use_second_order is True).
        self.model = model
        self.device = device
        self.beta1_schedule = beta1_schedule
        self.beta2 = beta2
        self.gamma = gamma
        self.lr_schedule = lr_schedule
        self.use_second_order = use_second_order
        self.entropy = entropy
        self.entropy_scheduler = entropy_scheduler
        # Initialize eligibility traces.
        self.z = [torch.zeros_like(p, device=device) for p in self.model.parameters()]

        if self.use_second_order:
            # Initialize second moment vectors (like Adam).
            self.v = [torch.zeros_like(p, device=device) for p in self.model.parameters()]
            self.it = 1

    # Reset the eligibility traces.
    def reset_z(self):
        self.z = [z.zero_() for z in self.z]

    # Accumulate gradients into the eligibility traces.
    def accumulate(self):
        self.z = [z.mul_(self.beta1_schedule.get_lr() * self.gamma).add_(p.grad) for z, p in zip(self.z, self.model.parameters())]

    # Perform an optimization step using the eligibility traces.
    def step(self, advantage, entropy):
        eps = 1e-8

        # Scale the eligibility traces by the advantage.
        z_hat = [z * (advantage) for z in self.z]
        if self.entropy:
            # Compute entropy gradient if entropy regularization is used.
            self.zero_grad()
            entropy.backward()

        if self.use_second_order:
            # Update second moment vectors (like Adam).
            self.v = [z.lerp(torch.square(g), self.beta2) for z, g in zip(self.v, z_hat)]
            v_hat = [v / (1 - self.beta2 ** self.it) for v in self.v]

            # Update model parameters using Adam-like update rule.
            for p, z, v in zip(self.model.parameters(), z_hat, v_hat):
                p.add_(self.lr_schedule.get_lr() / (torch.sqrt(v) + eps) * z)
            self.it += 1
        else:
            # Update model parameters using SGD-like update rule.
            for p, z in zip(self.model.parameters(), z_hat):
                term_to_add = z
                if self.entropy:
                    # Add entropy gradient if entropy regularization is used.
                    term_to_add += self.entropy_scheduler.get_lr() * p.grad
                p.add_(self.lr_schedule.get_lr() * term_to_add)

    # Zero the gradients of the model.
    def zero_grad(self):
        self.model.zero_grad()