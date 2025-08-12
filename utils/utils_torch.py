import math
import torch
from torch.optim import Optimizer
from torch import lerp
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
class TorchDeque():

    def __init__(self, maxlen, num_features,dtype, device):
        self.maxlen = maxlen
        self.memory = torch.empty((maxlen, num_features), dtype= dtype,device= device)  
        self.index = 0
        self.size = 0
        self.start = 0

    def fill(self, data):
        self.memory = data.repeat(self.maxlen,1)
        self.size = self.maxlen
        
    def push(self, data):
        if self.size == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            self.size += 1
        old = self.memory[self.index]
        self.memory[self.index] = data  
        self.index = (self.index + 1) % self.maxlen
        return old
    
    def sample(self, num_samples):
        indices = torch.randperm(torch.min(num_samples, self.size))[: num_samples]
        return self.memory[indices]
        
    def get_all_content_as_tensor(self):
        return torch.roll(self.memory, -self.start, dims= 0).flatten()
    
    def __sizeof__(self):
        return self.size
    
class Cascade_Memory():
    def __init__(self, memory_sizes, num_features, device):
        self.recent = TorchDeque(memory_sizes[0], num_features, torch.float32, device)
        self.intermediate = TorchDeque(memory_sizes[1], num_features, torch.float32, device)
        self.old = TorchDeque(memory_sizes[2], num_features, torch.float32, device)

    def push(self, data):
        x = self.recent.push(data)
        if x:
            x = self.intermediate.push(x)
        if x:
            x = self.old.push(x)
        
    def sample_recent(self, num_samples):
        return self.recent.sample(num_samples)
    
    def sample_old(self, num_samples):
        return self.old.sample(num_samples)
    

class CosineAnnealingWarmupLr(SequentialLR):

    def __init__(self, optimizer, warmup_steps, total_steps, start_factor=1e-3, last_epoch=-1, eta_min = 1e-6):
        self.warmup = LinearLR(optimizer, start_factor, 1, warmup_steps)
        self.cosineAnnealing = CosineAnnealingLR(optimizer, T_max= total_steps - warmup_steps, eta_min = eta_min)
        super().__init__(optimizer, [self.warmup, self.cosineAnnealing], [warmup_steps], last_epoch)


class CustomLrScheduler():
    def __init__(self):
        self.step = 0
    
    def get_lr(self):
        raise NotImplementedError
    
    def step_forward(self):
        self.step += 1


class CustomLrSchedulerCosineAnnealing(CustomLrScheduler):
    
    def __init__(self, base_lr, T_max, eta_min = 0):
        super().__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = base_lr

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min)* (1 + math.cos(math.pi * self.step / self.T_max))/ 2

class CustomLrSchedulerLinear(CustomLrScheduler):
    def __init__(self, initial_lr, end_lr, T_max):
        super().__init__()
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.T_max = T_max
    
    def get_lr(self):
        return self.initial_lr + (self.step)/(self.T_max) * (self.end_lr - self.initial_lr)
    
class CustomComposeSchedulers(CustomLrScheduler):
    def __init__(self, schedulers, milestones):
        super().__init__()
        self.current_idx = 0
        self.schedulers = schedulers
        self.milestones = milestones

    def get_lr(self):
        return self.schedulers[self.current_idx].get_lr()

    def step_forward(self):
        if self.step >= self.milestones[self.current_idx + 1]:
            self.current_idx += 1
    

class CustomWarmupCosineAnnealing(CustomComposeSchedulers):
    def __init__(self, initial_lr, max_lr, len_warmup, tot_len, eta_min):
        cosineAnnealing = CustomLrSchedulerCosineAnnealing(max_lr, tot_len - len_warmup, eta_min)
        warmupLr = CustomLrSchedulerLinear(initial_lr, max_lr, len_warmup)
        super().__init__([warmupLr, cosineAnnealing], [0, len_warmup])


class CustomAdamDuoEligibility():
    
    def __init__(self, actor, critic, device, lr_w_schedule, lr_theta_schedule, beta1_w_schedule, beta1_theta_schedule, entropy, entropy_scheduler, gamma,use_second_order = False, beta2 = 0.999):
        self.adam_theta = CustomAdamEligibility(actor, device, lr_theta_schedule, beta1_theta_schedule, entropy, entropy_scheduler, gamma, use_second_order, beta2)
        self.adam_w = CustomAdamEligibility(critic, device, lr_w_schedule, beta1_w_schedule, False, None, gamma, use_second_order, beta2)

    def reset_zw_ztheta(self):
        self.adam_theta.reset_z()
        self.adam_w.reset_z()

    def accumulate_and_step(self, advantage, entropy):
        self.adam_theta.accumulate()
        self.adam_theta.step(advantage, entropy)

        self.adam_w.accumulate()
        self.adam_w.step(advantage, entropy)

    def step(self, advantage, entropy):
        self.adam_theta.step(advantage, entropy)
        self.adam_w.step(advantage, entropy)

    def zero_grad(self):
        self.adam_theta.zero_grad()
        self.adam_w.zero_grad()
        

        

class CustomAdamEligibility():
    def __init__(self, model, device, lr_schedule, beta1_schedule, entropy, entropy_scheduler, gamma,use_second_order = False, beta2 = 0.999):
        self.model = model
        self.device = device
        self.beta1_schedule = beta1_schedule
        self.beta2 = beta2
        self.gamma = gamma
        self.lr_schedule = lr_schedule
        self.use_second_order = use_second_order
        self.entropy = entropy
        self.entropy_scheduler = entropy_scheduler
        self.z = [torch.zeros_like(p, device= device) for p in self.model.parameters()]
      
        if self.use_second_order:
            self.v = [torch.zeros_like(p, device= device) for p in  self.model.parameters()]
            self.it = 1
        
    def reset_z(self):
        self.z = [z.zero_() for z in self.z]

    def accumulate(self):
        self.z = [z.mul_(self.beta1_schedule.get_lr() * self.gamma).add_(p.grad) for z, p in zip(self.z, self.model.parameters())]

    def step(self, advantage, entropy):
        eps = 1e-8
       
        z_hat = [z * (advantage) for z in self.z]

        if self.entropy:
            self.zero_grad()
            entropy.backward()
        
        if self.use_second_order:
            self.v = [z.lerp(torch.square(g), self.beta2) for z, g in zip(self.v, z_hat)]

            v_hat = [v / (1 - self.beta2 ** self.it) for v in self.v]
        
            for p, z, v in zip(self.model.parameters(), z_hat, v_hat):
                p.add_(self.lr_schedule.get_lr()/ (torch.sqrt(v) + eps) * z)             
            self.it += 1
        else:
            for p, z in zip(self.model.parameters(), z_hat):
                term_to_add = z
                if self.entropy:
                    term_to_add += self.entropy_scheduler.get_lr() * p.grad 
                p.add_(self.lr_schedule.get_lr() * term_to_add)

    def zero_grad(self):
        self.model.zero_grad()
        

        


        