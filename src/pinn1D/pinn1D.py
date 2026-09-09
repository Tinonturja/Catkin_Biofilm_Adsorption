import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd



class KineticsPINNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, initial_qe):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        # Learnable Parmaters
        self.k2 = nn.Parameter(torch.tensor([[0.7]], dtype = torch.float32), requires_grad=True)
        self.qe = nn.Parameter(torch.tensor([[initial_qe]], dtype = torch.float32), requires_grad=True)

    def forward(self,x):
        return self.net(x)

class Isotherm