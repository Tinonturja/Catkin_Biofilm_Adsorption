import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pinn1D_train import isotherm_model, kinetics_model, device


# After training, check what Ce the model settled on
with torch.no_grad():
    Co_check = torch.tensor([[20.], [30.], [40.], [50.], [60.]], 
                             dtype=torch.float32).to(device)
    k_f = F.softplus(isotherm_model.k_f)
    inv_n = F.softplus(isotherm_model.inv_n)
    
    # Run predict_qe and inspect intermediate Ce
    qe = 0.5 * Co_check * 0.1  # starting guess
    for _ in range(40):
        Ce = torch.clamp(Co_check - qe/0.1, min=1e-8)
        qe_new = k_f * Ce.pow(inv_n)
        qe = 0.7 * qe + 0.3 * qe_new
    
    print("Co   | Ce (PINN) | Ce (mass-balance) | qe (PINN) | qe (actual)")
    Ce_mb = Co_check - torch.tensor([[0.74650],[1.13912],[1.55587],[2.06288],[2.48536]]).to(device) / 0.1
    for i in range(5):
        print(f"{Co_check[i,0]:.0f}   | {Ce[i,0]:.4f}    | {Ce_mb[i,0]:.4f}            | {qe[i,0]:.5f}   | {[0.74650,1.13912,1.55587,2.06288,2.48536][i]:.5f}")