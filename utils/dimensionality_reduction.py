import torch
import torch.nn as nn

## inspired from https://github.com/gngdb/pytorch-pca/blob/main/pca.py 
class PCA(nn.Module):
    def __init__(self, size_input, num_components,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_input = size_input
        self.num_components = num_components

    def fit(self, X):
        n, d = X.shape
        assert self.num_components <= d and self.num_components <= n, 'num components too big'
        mean = X.mean(dim = 0, keepdims = True)
        self.mean = mean
        Z = X - self.mean
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        self.Vh = Vh[:self.num_components,:]

    def forward(self, data):
        assert hasattr(self, 'Vh'), 'need to call fit'
        return (data - self.mean) @ self.Vh.T

