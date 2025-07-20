import torch
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


        

        

        