import torch, torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim,8), nn.ReLU(), nn.Linear(8,2))
        self.dec = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,input_dim))
    def forward(self,x): return self.dec(self.enc(x))
