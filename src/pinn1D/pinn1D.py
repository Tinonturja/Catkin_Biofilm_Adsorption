import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd

class PINN1DModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        # Learnable Parmaters
        self.k2 = nn.Parameter(torch.tensor(0.7), requires_grad=True)
        self.k_f = nn.Parameter(torch.tensor(-1.2),requires_grad=True)
        self.inv_n = nn.Parameter(torch.tensor(-2.5),requires_grad=True)
        self.qe = nn.Parameter(torch.tensor(5.0),requires_grad=True)

    def forward(self,x):
        return self.net(x)


    