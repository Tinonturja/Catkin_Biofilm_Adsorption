import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN3DModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,initial_k2:torch.tensor, initial_kf:torch.tensor, initial_n:torch.tensor):
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
        self.kf = nn.Parameter(data = initial_kf, requires_grad=True)
        self.n = nn.Parameter(data = initial_n, requires_grad=True)

    def forward(self, x):
        output = self.net(x)
        return output
    
def predict_qe(co, dosage, kf, n, n_iter:int = 40, damping:float = 0.3, return_diagnostics:bool = False):
    qe = 0.5 * co * dosage
    clamp_hits = 0
    for _ in range(n_iter):
        raw_ce = co - qe/dosage
        if return_diagnostics:
            clamp_hits += int((raw_ce <= 1e-8).sum().item())
        Ce = torch.clamp(raw_ce, min = 1e-8)
        qe_new = kf * (Ce**n)
        qe = (1-damping)* qe + damping * qe_new

    if return_diagnostics:
        diagnostics = {
            'qe_min': qe.min().item(),
            'qe_max': qe.max().item(),
            'qe_mean': qe.mean().item(),
            'clamp_hits': clamp_hits,
        }
        return qe, diagnostics
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