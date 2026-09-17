import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
# Prefer the qualified package import: src/pinn1D also has a bare-named
# data.py, and if that one gets imported first into sys.modules under the
# name "data" (e.g. from notebook cells that mix both packages), a plain
# `from data import ...` here would silently resolve to the wrong module.
try:
    from pinn3D.data import CombinedData, load_data
    from pinn3D.models import PINN3DModel, predict_qe
except ImportError:
    from data import CombinedData, load_data
    from models import PINN3DModel, predict_qe
import warnings
warnings.filterwarnings('ignore')
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'

EPOCHS = 8000
# PSO loss needs torch.autograd.grad(create_graph=True) through LayerNorm/Tanh
# (double backward); PyTorch's MPS backend produces NaN gradients for some
# inputs on this path, so training must run on CPU.
DEVICE = torch.device('cpu')
N_MIN = 0.05
N_MAX = 1.0
K2_FLOOR = 1e-3

# --- Fixed manual loss weights (loss-spike fix, see PINN3D_knowledge_transfer.md
# Section 4) ---
# Diagnosis: total_loss summed the four terms unweighted. PSO Loss reaches a
# steady-state floor (~4.3e-5) that is 10-180x LARGER than the other three
# terms' floors, while also being the flattest (lowest epoch-to-epoch
# variance) of the four. Because Adam's per-parameter adaptive scaling
# normalizes by each parameter's *combined* gradient variance (all four terms
# are summed before backward()), that persistently-large-but-steady PSO
# gradient was getting an outsized effective step relative to the noisier
# Data/Initial/ISO gradients, which share the same hidden representation --
# producing the observed ~10-epoch cycles where Data/Initial/ISO loss rise
# and fall together while PSO stays flat. This was confirmed to NOT be caused
# by predict_qe's fixed-point solve (K2/Kf/N moved smoothly through a spike,
# Clamp_hits stayed 0).
#
# Weights below bring all four terms to a comparable order of magnitude so no
# single term's gradient dominates the shared representation. Derived from
# the median of each raw loss column over the last 2000 epochs (steady-state
# region, post burn-in) of results/pinn3d_training_results.csv from the
# pre-fix run:
#   Data Loss median ~2.35e-7, PSO Loss median ~4.32e-5,
#   Initial Loss median ~1.20e-6, ISO Loss median ~5.18e-6
# weight_i = (PSO median) / (term_i median), i.e. PSO_LOSS_WEIGHT=1 is the
# reference scale and the other three are scaled up to match it. Re-derive
# these (same formula, against a fresh training_results.csv) if the loss
# floors shift meaningfully after retraining with these weights in place --
# they are a starting point, not a permanent constant.
DATA_LOSS_WEIGHT = 180.0
PSO_LOSS_WEIGHT = 1.0
INITIAL_LOSS_WEIGHT = 35.0
ISO_LOSS_WEIGHT = 8.0

def transform_k2(raw_k2):
    return F.softplus(raw_k2) + K2_FLOOR

def transform_n(raw_n):
    return N_MIN + (N_MAX - N_MIN) * torch.sigmoid(raw_n)

def training_pinn3d(data:CombinedData,
                    model:PINN3DModel,
                    optimizer:torch.optim.Optimizer,
                    loss_fn:torch.nn.Module,
                    epochs:int = EPOCHS,
                    device = DEVICE):
    result_dict = {
        'Data Loss':[],
        'PSO Loss':[],
        'Initial Loss':[],
        'ISO Loss':[],
        'Weighted Total Loss':[],
        # --- diagnostics for the loss-spike investigation ---
        'K2':[],
        'Kf':[],
        'N':[],
        'Qe_min':[],
        'Qe_max':[],
        'Qe_mean':[],
        'Clamp_hits':[],
    }
    model.train()
    model.to(device)
    # get the input, output and collocation data
    co = data.collocated_input[:,1:2]
    co = torch.tensor(co, dtype = torch.float32, device = device)
    dosage = data.collocated_input[:,2:3]
    dosage = torch.tensor(dosage, dtype = torch.float32, device = device)
    # pass the data into the allocated device
    data.pass_to_device([data.scaled_input_tensor,data.scaled_output_tensor,data.collocated_scaled_input_tensor,
                        data.initial_scaled_datapoints_tensor,
                        data.max_time_tensor])
    # The model is trained to output STANDARDIZED adsorption (matched against
    # scaled_output_tensor in data_loss). qe_real from predict_qe() and the
    # initial/equilibrium physics targets are all in REAL mg/g units, so every
    # model output touching them has to be unscaled first, or the physics
    # losses are enforced against the wrong absolute scale and offset.
    output_mean = torch.tensor(data.output_mean, dtype = torch.float32, device = device)
    output_std = torch.tensor(data.output_std, dtype = torch.float32, device = device)

    def unscale_output(scaled_output):
        return scaled_output * output_std + output_mean

    for epoch in range(epochs):
        optimizer.zero_grad()
        # pass the input to the model
        pred_output = model(data.scaled_input_tensor)
        # get the loss
        data_loss = loss_fn(pred_output, data.scaled_output_tensor.view(-1,1))
        result_dict['Data Loss'].append(data_loss.item())
        # pso loss
        collocation_pred_adsorption = model(data.collocated_scaled_input_tensor)
        # scale the learnable parameters
        positive_k2 = transform_k2(model.k2)
        positive_kf = F.softplus(model.kf)
        positive_n = transform_n(model.n)
        qe_real, qe_diag = predict_qe(co = co,
                                dosage = dosage,
                                kf = positive_kf,
                                n = positive_n,
                                return_diagnostics = True)
        qe_real.to(device)
        # log the physical-parameter / fixed-point diagnostics every epoch --
        # this is what we're using to confirm the loss-spike hypothesis:
        # a jump in Qe_min/Qe_max/Clamp_hits should line up with the spikes.
        result_dict['K2'].append(positive_k2.item())
        result_dict['Kf'].append(positive_kf.item())
        result_dict['N'].append(positive_n.item())
        result_dict['Qe_min'].append(qe_diag['qe_min'])
        result_dict['Qe_max'].append(qe_diag['qe_max'])
        result_dict['Qe_mean'].append(qe_diag['qe_mean'])
        result_dict['Clamp_hits'].append(qe_diag['clamp_hits'])
        dq_dt_n = torch.autograd.grad(
            outputs = collocation_pred_adsorption.sum(),
            inputs = data.collocated_scaled_input_tensor,
            create_graph = True
        )[0][:, 0:1] # dq/dt, dq/dc, dq/ddosage
        # dq/dt = k2*(qe-qt)**2
        # dq_dt_n is d(scaled_output)/d(scaled_time). Chain rule to get
        # d(real_output)/d(real_time): multiply by d(scaled_time)/d(real_time)
        # = 1/t_std, AND by d(real_output)/d(scaled_output) = output_std.
        dq_dt_real = dq_dt_n * output_std * (1/data.input_stats['Time']['std'])
        collocation_pred_adsorption_real = unscale_output(collocation_pred_adsorption)
        pso_loss = torch.mean((dq_dt_real - (positive_k2 * (qe_real - collocation_pred_adsorption_real)**2))**2)
        result_dict['PSO Loss'].append(pso_loss.item())
        # initial loss (real adsorption must be 0 at t=0, not the scaled output)
        initial_adsorption_pred = model(data.initial_scaled_datapoints_tensor)
        initial_adsorption_pred_real = unscale_output(initial_adsorption_pred)
        initial_loss = torch.mean(initial_adsorption_pred_real**2)
        result_dict['Initial Loss'].append(initial_loss.item())
        # isotherm loss (Using Freundlich equation)
        # Ce = Co - qe/vm
        # qe = k_f * Ce**n
        # iso loss --> # qe_real: the max adsorption , collocation_pred_adsorption:
        max_adsorption_pred = model(data.max_time_tensor)
        max_adsorption_pred_real = unscale_output(max_adsorption_pred)
        iso_loss = torch.mean((qe_real - max_adsorption_pred_real)**2)
        result_dict['ISO Loss'].append(iso_loss.item())
        # weighted sum (see DATA_LOSS_WEIGHT et al. above) -- this is what
        # actually gets backpropagated; the four raw losses logged above stay
        # unweighted so they're directly comparable to the pre-fix run.
        total_loss = (DATA_LOSS_WEIGHT * data_loss
                      + PSO_LOSS_WEIGHT * pso_loss
                      + INITIAL_LOSS_WEIGHT * initial_loss
                      + ISO_LOSS_WEIGHT * iso_loss)
        result_dict['Weighted Total Loss'].append(total_loss.item())
        total_loss.backward()
        optimizer.step()

        # print
        if (epoch+1)%100 == 0:
            flag = f" | !! clamp_hits={qe_diag['clamp_hits']} !!" if qe_diag['clamp_hits'] > 0 else ""
            print(f"Epoch:{epoch+1}/{epochs} | Data Loss: {data_loss.item():.3f} | PSO Loss: {pso_loss.item():.3f} | Initial Loss: {initial_loss.item():.4f} | ISO Loss: {iso_loss.item():.4f} | Weighted Total Loss: {total_loss.item():.4f} | Kf: {positive_kf.item():.4f} | N: {positive_n.item():.4f} | K2: {positive_k2.item():.4f} | Qe: [{qe_diag['qe_min']:.4f}, {qe_diag['qe_max']:.4f}]{flag}")
    # make the result a dictionary
    print('-'*120)
    print(f"Learnable Parameters")
    print('-'*120)
    print(f"Rate Constant (K2):{transform_k2(model.k2).item()}")
    print(f"Freundlich Constant (Kf):{F.softplus(model.kf)}")
    print(f"N:{transform_n(model.n)}")
    print(f"Total Loss:")
    result_df = pd.DataFrame(result_dict)
    result_df.insert(0, 'Epoch', range(1, len(result_df) + 1))
    result_df['Total Loss'] = result_df[['Data Loss', 'PSO Loss', 'Initial Loss', 'ISO Loss']].sum(axis=1)
    return result_df

def save_training_results(result_df:pd.DataFrame, path=RESULTS_DIR / 'pinn3d_training_results.csv'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path, index=False)
    return path

def save_parameters(model:PINN3DModel, result_df:pd.DataFrame, path=RESULTS_DIR / 'pinn3d_parameters.md'):
    """Write the learned physical parameters and final loss values to a markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k2 = transform_k2(model.k2).item()
    kf = F.softplus(model.kf).item()
    n = transform_n(model.n).item()
    final_row = result_df.iloc[-1]
    lines = [
        "# PINN3D Learned Parameters",
        "",
        f"- Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Epochs: {len(result_df)}",
        "",
        "## Physical Parameters",
        "",
        f"- Rate Constant (K2): {k2:.6f}",
        f"- Freundlich Constant (Kf): {kf:.6f}",
        f"- Freundlich Exponent (N): {n:.6f}",
        "",
        "## Final Losses",
        "",
        f"- Data Loss: {final_row['Data Loss']:.6f}",
        f"- PSO Loss: {final_row['PSO Loss']:.6f}",
        f"- Initial Loss: {final_row['Initial Loss']:.6f}",
        f"- ISO Loss: {final_row['ISO Loss']:.6f}",
        f"- Total Loss: {final_row['Total Loss']:.6f}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    data_path = PROJECT_ROOT / 'data' / 'data_of_biofilm.xlsx'
    data = load_data(data_path)
    INPUT_SIZE = 3
    HIDDEN_SIZE  = 32
    OUTPUT_SIZE = 1
    INITIAL_K2 = torch.tensor([-3.0], dtype = torch.float32)
    INITIAL_KF = torch.tensor([-1.0], dtype = torch.float32)
    # raw input to transform_n (bounded sigmoid, not softplus) -> n starts at ~0.52
    INITIAL_N = torch.tensor([0.0], dtype = torch.float32)
    model = PINN3DModel(input_size = INPUT_SIZE,
                        hidden_size = HIDDEN_SIZE,
                        output_size = OUTPUT_SIZE,
                        initial_k2 = INITIAL_K2,
                        initial_kf = INITIAL_KF,
                        initial_n = INITIAL_N)
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr = 0.005)
    LOSS_FN = torch.nn.MSELoss()
    start_time = time.time()
    print("="* 120)
    print("Training Start")
    print("="* 120)
    result_df = training_pinn3d(data = data,
    model = model,
    optimizer = OPTIMIZER,
    loss_fn = LOSS_FN,
    )
    print("="* 120)
    print("Training End")
    print("="* 120)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time recorded to train: {total_time}")
    results_path = save_training_results(result_df)
    parameters_path = save_parameters(model, result_df)
    print(f"Saved training loss history to: {results_path}")
    print(f"Saved learned parameters to: {parameters_path}")
