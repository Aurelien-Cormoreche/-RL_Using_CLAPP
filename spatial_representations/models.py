import torch.nn as nn

class Spatial_Model(nn.Module):
    def __init__(self,input_dim, dimensions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim
        self.dimensions = dimensions
        self.layers = []
        prev_dim = input_dim
        
        for dim in dimensions:
            #self.layers.append(nn.LayerNorm(prev_dim))
            self.layers.append(nn.Dropout(p = 0.0))
            self.layers.append(nn.Linear(prev_dim, dim))
            self.layers.append(nn.GELU())
            prev_dim = dim
        self.layers.pop()
        print(self.layers)
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.layers(x)