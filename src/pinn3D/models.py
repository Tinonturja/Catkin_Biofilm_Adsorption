import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN3DModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,initial_k2:torch.tensor, initial_qe:torch.tensor, initial_kf:torch.tensor, initial_n:torch.tensor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size,
                      out_features = hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        # Learnable Parameters
        self.k2 = nn.Parameter(data = initial_k2, requires_grad=True)
        self.qe = nn.Parameter(data = initial_qe, requires_grad=True)
        self.kf = nn.Parameter(data = initial_kf, requires_grad=True)
        self.n = nn.Parameter(data = initial_n, requires_grad=True)

    def forward(self, x):
        output = self.net(x)
        return output
    
    def predict_qe(self, co, dosage, n_iter:int = 40, damping:float = 0.3):
        qe = 0.5 * co * dosage 
        for _ in range(n_iter):
            Ce = torch.clamp(co - qe/dosage, min = 1e-8)
            qe_new = self.k2 * (Ce**self.n)
            qe = (1-damping)* qe + damping * qe
            return qe
"""
if __name__ == "__main__":
    initial_k2 = torch.tensor([[-2.0]])
    initial_qe = torch.tensor([[1.25]])
    initial_kf = torch.tensor([[0.673]])
    initial_n = torch.tensor([[0.23]])
    input_size = 3
    hidden_size = 16
    output_size = 1
    pinn3D_model = PINN3DModel(input_size = input_size,
                               hidden_size = hidden_size,
                               output_size = output_size,
                               initial_k2= initial_k2,
                               initial_qe = initial_qe,
                               initial_kf = initial_kf,
                               initial_n = initial_n)
    print(pinn3D_model.state_dict())
    
"""