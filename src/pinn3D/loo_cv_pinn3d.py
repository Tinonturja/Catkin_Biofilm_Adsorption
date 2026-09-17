import sys, time, json, argparse, os
from pathlib import Path
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from pinn3D.data import CombinedData, DataConfig
from pinn3D.models import PINN3DModel, predict_qe
from pinn3D.train import transform_k2, transform_n, DATA_LOSS_WEIGHT, PSO_LOSS_WEIGHT, INITIAL_LOSS_WEIGHT, ISO_LOSS_WEIGHT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(PROJECT_ROOT / 'data' / 'data_of_biofilm.xlsx')
OUT_DIR = str(PROJECT_ROOT / 'results' / 'loo_cv')
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
LR = 0.005


def build_combined_df():
    config = DataConfig()
    data = CombinedData(data_path=DATA_PATH, config=config)
    data.process_kinetics_data()
    data.process_isotherm_data()
    data.combined_data()
    combined_df = data.combined_df.reset_index(drop=True)
    combined_df['arm'] = ['kinetics'] * len(data.kinetics_data_df) + ['isotherm'] * len(data.isotherm_data_df)
    return combined_df, config


def prepare_fold(combined_df, holdout_idx, config):
    train_df = combined_df.drop(index=holdout_idx).reset_index(drop=True)
    holdout_row = combined_df.loc[holdout_idx]

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    X_train = train_df[list(config.input_columns)]
    y_train = train_df[config.target_column]
    scaled_input = input_scaler.fit_transform(X_train)
    scaled_output = output_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

    scaled_input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    scaled_output_tensor = torch.tensor(scaled_output, dtype=torch.float32)

    input_mean = input_scaler.mean_
    input_std = input_scaler.scale_
    output_mean = output_scaler.mean_
    output_std = output_scaler.scale_
    input_stats = {f: {'mean': m, 'std': s} for f, m, s in zip(config.input_columns, input_mean, input_std)}

    time_alloc = np.linspace(0, 200, num=config.time_collocation_num)
    vm_alloc = np.linspace(0.030, 0.120, num=config.vm_collocation_num)
    conc_alloc = np.linspace(15, 65, num=config.conc_collocation_num)
    T, C, V = np.meshgrid(time_alloc, conc_alloc, vm_alloc)
    collocated_input = np.column_stack([T.reshape(-1, 1), C.reshape(-1, 1), V.reshape(-1, 1)])
    collocated_scaled = input_scaler.transform(collocated_input)
    collocated_scaled_input_tensor = torch.tensor(collocated_scaled, dtype=torch.float32, requires_grad=True)

    max_time_scaled = (config.t_max - input_stats['Time']['mean']) / input_stats['Time']['std']
    collocated_maxtime_scaled = collocated_scaled.copy()
    collocated_maxtime_scaled[:, 0] = max_time_scaled
    max_time_tensor = torch.tensor(collocated_maxtime_scaled, dtype=torch.float32)

    initial_time_datapoints = collocated_scaled.copy()
    initial_t_scaled = (0 - input_stats['Time']['mean']) / input_stats['Time']['std']
    initial_time_datapoints[:, 0:1] = initial_t_scaled
    initial_scaled_datapoints_tensor = torch.tensor(initial_time_datapoints, dtype=torch.float32)

    holdout_input = holdout_row[list(config.input_columns)].to_numpy().reshape(1, -1).astype(float)
    holdout_scaled_input = input_scaler.transform(holdout_input)
    holdout_input_tensor = torch.tensor(holdout_scaled_input, dtype=torch.float32)
    holdout_actual = float(holdout_row[config.target_column])

    return dict(
        scaled_input_tensor=scaled_input_tensor,
        scaled_output_tensor=scaled_output_tensor,
        collocated_input=collocated_input,
        collocated_scaled_input_tensor=collocated_scaled_input_tensor,
        max_time_tensor=max_time_tensor,
        initial_scaled_datapoints_tensor=initial_scaled_datapoints_tensor,
        input_stats=input_stats,
        output_mean=output_mean, output_std=output_std,
        holdout_input_tensor=holdout_input_tensor,
        holdout_actual=holdout_actual,
        arm=holdout_row['arm'],
    )


def train_fold(fold, epochs, seed=SEED, lr=LR, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PINN3DModel(input_size=3, hidden_size=32, output_size=1,
                         initial_k2=torch.tensor([-3.0]), initial_kf=torch.tensor([-1.0]),
                         initial_n=torch.tensor([0.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    co = torch.tensor(fold['collocated_input'][:, 1:2], dtype=torch.float32)
    dosage = torch.tensor(fold['collocated_input'][:, 2:3], dtype=torch.float32)
    output_mean_t = torch.tensor(fold['output_mean'], dtype=torch.float32)
    output_std_t = torch.tensor(fold['output_std'], dtype=torch.float32)

    def unscale(x):
        return x * output_std_t + output_mean_t

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_output = model(fold['scaled_input_tensor'])
        data_loss = loss_fn(pred_output, fold['scaled_output_tensor'].view(-1, 1))

        collocation_pred = model(fold['collocated_scaled_input_tensor'])
        positive_k2 = transform_k2(model.k2)
        positive_kf = F.softplus(model.kf)
        positive_n = transform_n(model.n)
        qe_real = predict_qe(co=co, dosage=dosage, kf=positive_kf, n=positive_n)

        dq_dt_n = torch.autograd.grad(outputs=collocation_pred.sum(),
                                       inputs=fold['collocated_scaled_input_tensor'],
                                       create_graph=True)[0][:, 0:1]
        dq_dt_real = dq_dt_n * output_std_t * (1 / fold['input_stats']['Time']['std'])
        collocation_pred_real = unscale(collocation_pred)
        pso_loss = torch.mean((dq_dt_real - (positive_k2 * (qe_real - collocation_pred_real) ** 2)) ** 2)

        initial_pred = model(fold['initial_scaled_datapoints_tensor'])
        initial_pred_real = unscale(initial_pred)
        initial_loss = torch.mean(initial_pred_real ** 2)

        max_pred = model(fold['max_time_tensor'])
        max_pred_real = unscale(max_pred)
        iso_loss = torch.mean((qe_real - max_pred_real) ** 2)

        total_loss = (DATA_LOSS_WEIGHT * data_loss + PSO_LOSS_WEIGHT * pso_loss
                      + INITIAL_LOSS_WEIGHT * initial_loss + ISO_LOSS_WEIGHT * iso_loss)
        total_loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 1000 == 0:
            print(f"  epoch {epoch+1}/{epochs} data_loss={data_loss.item():.5f} total={total_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_scaled = model(fold['holdout_input_tensor']).item()
    pred_real = pred_scaled * fold['output_std'][0] + fold['output_mean'][0]
    return pred_real


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=str, required=True, help='comma-separated fold indices, or "all"')
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--out', type=str, default=f'{OUT_DIR}/loo_results.json')
    args = ap.parse_args()

    combined_df, config = build_combined_df()
    n_total = len(combined_df)
    if args.folds == 'all':
        fold_indices = list(range(n_total))
    else:
        fold_indices = [int(x) for x in args.folds.split(',')]

    # load existing results if present, to accumulate across multiple tool calls
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    else:
        results = {}

    for idx in fold_indices:
        t0 = time.time()
        fold = prepare_fold(combined_df, idx, config)
        pred = train_fold(fold, epochs=args.epochs)
        dt = time.time() - t0
        results[str(idx)] = {
            'arm': fold['arm'],
            'actual': fold['holdout_actual'],
            'predicted': float(pred),
            'time_sec': dt,
        }
        print(f"fold {idx} ({fold['arm']}): actual={fold['holdout_actual']:.5f} pred={pred:.5f} "
              f"abs_err={abs(pred-fold['holdout_actual']):.5f} time={dt:.1f}s")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)}/{n_total} folds to {args.out}")
