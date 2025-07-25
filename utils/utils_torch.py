import math
import torch
from torch.optim import Optimizer
from torch import lerp
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
class TorchDeque:

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
        self.memory[self.index] = data  
        self.index = (self.index + 1) % self.maxlen
        
    def get_all_content_as_tensor(self):
        return torch.roll(self.memory, -self.start, dims= 0).flatten()
    
    def __sizeof__(self):
        return self.size
    

class CosineAnnealingWarmupLr(SequentialLR):

    def __init__(self, optimizer, warmup_steps, total_steps, start_factor=1e-3, last_epoch=-1, eta_min = 1e-5):
        self.warmup = LinearLR(optimizer, start_factor, warmup_steps)
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
        step =  super().step()
        if step >= self.milestones[self.current_idx + 1]:
            self.current_idx += 1
        return step
    

class CustomWarmupCosineAnnealing(CustomComposeSchedulers):
    def __init__(self, initial_lr, max_lr, len_warmup, tot_len, eta_min):
        cosineAnnealing = CustomLrSchedulerCosineAnnealing(max_lr, tot_len - len_warmup, eta_min)
        warmupLr = CustomLrSchedulerLinear(initial_lr, max_lr, len_warmup)
        super().__init__([warmupLr, cosineAnnealing], [0, len_warmup])


class CustomAdamEligibility():
    
    def __init__(self, actor, critic, device, lr_w_schedule, lr_theta_schedule, beta1_w_schedule, beta1_theta_schedule, gamma,use_second_order = False, beta2 = 0.999):
        self.actor = actor
        self.critic = critic
        self.device = device
        self.beta1_w_schedule = beta1_w_schedule
        self.beta1_theta_schedule= beta1_theta_schedule
        self.beta2 = beta2
        self.gamma = gamma
        self.lr_w_schedule = lr_w_schedule
        self.lr_theta_schedule = lr_theta_schedule
        self.use_second_order = use_second_order
        self.z_w = [torch.zeros_like(p, device= device) for p in self.critic.parameters()]
        self.z_theta = [torch.zeros_like(p, device= device) for p in  self.actor.parameters()]

        if self.use_second_order:
            self.v_w = [torch.zeros_like(p, device= device) for p in  self.critic.parameters()]
            self.v_theta = [torch.zeros_like(p, device= device) for p in self.actor.parameters()]
            self.it = 1

    def reset_zw_ztheta(self):

        self.z_w = [z.zero_() for z in self.z_w]
        self.z_theta = [z.zero_() for z in self.z_theta]

    def step(self, advantage):

        eps = 1e-8

        self.z_w = [z.mul_(self.beta1_w_schedule.get_lr() * self.gamma).add_(p.grad) for z, p in zip(self.z_w, self.critic.parameters())]
        self.z_theta = [z.mul_(self.beta1_theta_schedule.get_lr() * self.gamma).add_(p.grad)  for z, p in zip(self.z_theta, self.actor.parameters())]

        z_w_hat = [z * (advantage) for z in self.z_w]
        z_theta_hat = [z * (advantage) for z in self.z_theta]
        
        if self.use_second_order:
            self.v_w = [z.lerp(torch.square(g), self.beta2) for z, g in zip(self.v_w, z_w_hat)]
            self.v_theta = [z.lerp(torch.square(g), self.beta2) for z, g in zip(self.v_theta, z_theta_hat)]

            v_w_hat = [v / (1 - self.beta2 ** self.it) for v in self.v_w]
            v_theta_hat = [v / (1 - self.beta2 ** self.it) for v in self.v_theta]

            for p, z, v in zip(self.critic.parameters(), z_w_hat, v_w_hat):
                p.add_(self.lr_w_schedule.get_lr()/ (torch.sqrt(v) + eps) * z)             
            for p, z, v in zip( self.actor.parameters(), z_theta_hat, v_theta_hat):    
                p.add_(self.lr_theta_schedule.get_lr()/ (torch.sqrt(v) + eps) * z)

            self.it += 1
        else:
            for p, z in zip(self.critic.parameters(), z_w_hat):
                p.add_(self.lr_w_schedule.get_lr() * z)              
            for p, z in zip( self.actor.parameters(), z_theta_hat):    
                p.add_(self.lr_theta_schedule.get_lr() * z)

    def zero_grad(self):
        self.actor.zero_grad()
        self.critic.zero_grad()
        

        




        
