import torch
import
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pinn1D_train
from data import Data
from pinn1D import KineticsPINNModel
import pandas as pd
from sklearn.metrics import r2_score

# After training, check what Ce the model settled on
def check_Ce_for_given_Co():
    with torch.no_grad():
        Co_check = torch.tensor([[20.], [30.], [40.], [50.], [60.]], 
                                 dtype=torch.float32).to(pinn1D_train.device)
        k2 = F.softplus(pinn1D_train.model.k2)
        qe = F.softplus(pinn1D_train.model.qe)
        
        # Run predict_qt and inspect intermediate Ce
        qt = 0.5 * Co_check * 0.1  # starting guess
        for _ in range(40):
            Ce = torch.clamp(Co_check - qt/0.1, min=1e-8)
            qt_new = qe - (1/k2) * (qt**2)
            qt = 0.7 * qt + 0.3 * qt_new
        
        print("Co   | Ce (PINN) | Ce (mass-balance) | qt (PINN) | qt (actual)")
        Ce_mb = Co_check - torch.tensor([[0.74650],[1.13912],[1.55587],[2.06288],[2.48536]]).to(pinn1D_train.device) / 0.1
        for i in range(5):
            print(f"{Co_check[i,0]:.0f}   | {Ce[i,0]:.4f}    | {Ce_mb[i,0]:.4f}            | {qt[i,0]:.5f}   | {[0.74650,1.13912,1.55587,2.06288,2.48536][i]:.5f}")

def adsorption_prediction(model, input_std, input_mean, collocation_input, device):
    """
    Predict the adsorption qt for given collocation input using the trained model
    """
    model.eval()
    model.to(device)
    collocation_input = torch.tensor(collocation_input, dtype = torch.float32, device = device).view(-1,1) 
    # scaled the input
    collocation_scaled_input = (collocation_input - input_mean)/input_std
    with torch.no_grad():
        qt_pred = model(collocation_scaled_input)
    return qt_pred
def calculate_r2(pred_qe, real_qe):
    pred_qe = pred_qe.detach().cpu().reshape(-1,1)
    real_qe = real_qe.reshape(-1,1)
    r2_value = r2_score(y_true = real_qe, 
                        y_pred = pred_qe)
    return r2_value
if __name__ == "__main__":
    # Load the trained model
    data_path = '/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/data of biofilm(TINON BHAI).xlsx'
    data = Data(data_path=data_path, dataset_name='kinetics').input_output_collocation_data(input_features='Time', output_feature= 'qt( catkin)', collocation_points=50)    
    data.processed_tensors()
    model = KineticsPINNModel(input_size=1, hidden_size=20, output_size=1, initial_qe=data.y_values.max() * 1.2)
    model.load_state_dict(torch.load('/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/models/kinetics_pinn_model.pth'))
    input_mean = data.input_mean_torch
    input_std = data.input_std_torch
    input_mean, input_std = input_mean.to(pinn1D_train.device), input_std.to(pinn1D_train.device)
    X, y = data.X_values, data.y_values.reshape(-1,1)
    collocation_input = data.collocation_input
    extended_time =np.linspace(0,300, num = 100)
    qt_pred = adsorption_prediction(model = model, input_std=input_std, input_mean = input_mean, collocation_input=collocation_input, device=pinn1D_train.device)
    qt_extended_pred = adsorption_prediction(model = model, input_std=input_std, input_mean = input_mean, collocation_input=extended_time, device=pinn1D_train.device)
    actual_data_prediction = adsorption_prediction(model = model, input_std=input_std, input_mean = input_mean, collocation_input=X, device=pinn1D_train.device)
    R2_Score = calculate_r2(pred_qe=actual_data_prediction,
                            real_qe=y )
    # Create a dictionary to store the results
    results_dict = {
        'Time (min)': np.round(collocation_input, 2),
        'Adsorption (mg/g)': np.round(qt_pred.detach().cpu().numpy().flatten(),2)
    }
    extended_results_dict = {
        'Time (min)': np.round(extended_time, 2),
        'Adsorption (mg/g)': np.round(qt_extended_pred.detach().cpu().numpy().flatten(),2)

    }
    
    results_df = pd.DataFrame(results_dict)
    extended_df = pd.DataFrame(extended_results_dict)
    results_df.to_csv('/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/results/kinetics_pinn_predictions.csv', index=False)
    extended_df.to_csv('/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/results/kinetics_pinn_extended_predictions.csv', index=False)
    print(R2_Score)