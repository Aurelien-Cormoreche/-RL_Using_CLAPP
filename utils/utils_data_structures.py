import torch

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
    

    
    
    

    

        

        

        