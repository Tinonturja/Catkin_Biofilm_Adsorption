import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from pinn1D import KineticsPINNModel
from pinn1D_train import device
from data import Data
from testing import adsorption_prediction

palette = {
    "data": "#3366CC",
    "pinn": "#E07A5F",
    "reference": "#3D9970",
}

# load the save model from the pinn1D_train.py file
def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    model.eval()

# Create a figure directory where all the figures will be saved in png format
def create_figure_directory(figure_directory):
    directory = os.fspath(figure_directory) if figure_directory is not None else './figures'
    os.makedirs(directory, exist_ok=True)
    return directory


def _as_numpy_1d(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().reshape(-1)
    if isinstance(values, np.ndarray):
        return values.reshape(-1)
    return np.asarray(values).reshape(-1)


def plot_adsorption(real_time, real_adsorption, collocation_time, collocation_qt, save_path, figure_directory='./figures'):
    """
    Left panel: real experimental data (scatter).
    Right panel: PINN predictions at collocation points (already computed correctly upstream).
    """
    figure_directory = create_figure_directory(figure_directory)
    real_time_arr = _as_numpy_1d(real_time)
    real_ads_arr = _as_numpy_1d(real_adsorption)
    coll_time_arr = _as_numpy_1d(collocation_time)
    coll_qt_arr = _as_numpy_1d(collocation_qt)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    axes[0].plot(real_time_arr, real_ads_arr, 'o', label='Experimental Data', color=palette['data'])
    axes[0].set_title("Adsorption Kinetics: Experimental Data")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Catkin adsorption (qt)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(coll_time_arr, coll_qt_arr, '-', linewidth=2, label='Model Prediction', color=palette['pinn'])
    axes[1].set_title('Adsorption Kinetics: Model Prediction')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('catkin adsorption (qt)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.savefig(os.path.join(figure_directory, save_path), dpi=300)
    plt.close()
if __name__ == "__main__":
    data_path = '/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/data of biofilm(TINON BHAI).xlsx'
    data = Data(data_path=data_path, dataset_name='kinetics').input_output_collocation_data(input_features='Time', output_feature= 'qt( catkin)', collocation_points=50)
    data.processed_tensors()
    initial_qe = data.y_values.max() * 1.2
    collocation_input = data.collocated_scaled_tensor
    real_time = data.X_values
    real_adsorption = data.y_values
    X = collocation_input * data.input_std_torch + data.input_mean_torch  # Scale back to original time values
    model = KineticsPINNModel(input_size=1, hidden_size=20, output_size=1, initial_qe=initial_qe)
    load_model(model, './models/kinetics_pinn_model.pth')
    predicted_adsorption = adsorption_prediction(model = model,
                                                 collocation_input=collocation_input,
                                                 device = device)
    figure_directory = create_figure_directory('./figures')
    save_path = 'adsorption_collocation_plot.png'
    plot_adsorption(
    real_time=real_time,
    real_adsorption=real_adsorption,
    collocation_time=X,
    collocation_qt=predicted_adsorption,
    save_path=save_path,
    figure_directory=figure_directory
)