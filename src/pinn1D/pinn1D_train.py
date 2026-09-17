## now the toughest part
## we will get the input data, output data and collocation points for both the datasets
# Create the model for the isotherm dataset and the kinetics dataset
import pandas as pd
import os
import torch
from data import Data
from pinn1D import KineticsPINNModel
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# src/pinn1D/pinn1D_train.py -> project root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# declare the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_kinetics_data(model,
                        optimizer,
                        loss_fn,
                        kinetics_data,
                        device = device,
                        num_epochs = 5000):
    results_dict = {'Data Loss': [], 'PSO Loss': [], 'Initial Loss': [], 'Overshoot Loss': []}
    model.train()
    model.to(device)
    kinetics_input = kinetics_data.X_scaled_tensor
    kinetics_output = kinetics_data.y_tensor
    kinetics_collocation = kinetics_data.collocated_scaled_tensor
    # pass the input and output data to the device
    kinetics_input = kinetics_input.to(device)
    kinetics_output = kinetics_output.to(device)
    kinetics_collocation = kinetics_collocation.to(device)
    t_std = kinetics_data.input_std_torch.to(device)
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass for kinetics data
        #==================================
        # LOSS A: DATA LOSS
        #==================================
        kinetics_pred = model(kinetics_input)
        kinetics_loss = loss_fn(kinetics_pred, kinetics_output)
        # Do the prediction on the collocation points
        #==================================
        # LOSS B: PSO LOSS
        #==================================
        collocation_prediction = model(kinetics_collocation)
        qe = F.softplus(model.qe)
        k2 = F.softplus(model.k2)
        """
        dq/dt = k2 * (qe-qt)**2
        dq/dt - (k2 * (qe-qt)**2) = 0
        """
        dq_dt_n = torch.autograd.grad(
            outputs = collocation_prediction,
            inputs = kinetics_collocation,
            grad_outputs=torch.ones_like(collocation_prediction),
            create_graph=True
        )[0]
        dq_dt_real = dq_dt_n * (1/t_std) # scale the derivative to the original time scale
        residual = dq_dt_real - (k2 * (qe - collocation_prediction) ** 2)
        # calculate the pso loss
        pso_loss = torch.mean(residual**2)
        #==================================
        # LOSS C: INITIAL LOSS
        #==================================            
        initial_scaled_time = kinetics_data.input_scaler_fn.transform(np.zeros(shape = (1,1)))
        initial_scaled_time_torch = torch.tensor(initial_scaled_time,dtype = torch.float32, device=device)
        initial_qt = model(initial_scaled_time_torch)
        initial_loss = torch.mean(initial_qt**2)
        #==================================
        # LOSS D: Overshoot LOSS
        #================================== 
        overshoot_loss = torch.mean(torch.relu(collocation_prediction - qe) ** 2)
        overshoot_weight = 5
        total_loss = kinetics_loss + pso_loss + initial_loss + overshoot_weight * overshoot_loss
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Kinetics Training:\nEpoch [{epoch + 1}/{num_epochs}] | Data Loss: {kinetics_loss.item():.4f} | PSO Loss: {pso_loss.item():.4f} | Initial Loss: {initial_loss.item():.4f} | Overshoot Loss: {overshoot_loss.item():.4f} | Total Loss: {total_loss.item():.4f}')
            results_dict['Data Loss'].append(kinetics_loss.item())
            results_dict['PSO Loss'].append(pso_loss.item())
            results_dict['Initial Loss'].append(initial_loss.item())
            results_dict['Overshoot Loss'].append(overshoot_loss.item())
    final_k2 = F.softplus(model.k2).item()
    final_qe = F.softplus(model.qe).item()
    # print the learnable parameters
    print("="* 60 )
    print("Training Completed!")
    print("="* 60 )
    print(f"Learnable Parameters:\n k2: {final_k2:.4f} | qe: {final_qe:.4f}")
    return results_dict

def save_the_model(model, save_path):

    """
    Save the model to the given path
    """
    # create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    # save the result dictionary
# get the data

if __name__ == "__main__":
    data_path = str(PROJECT_ROOT / 'data' / 'data_of_biofilm.xlsx')
    data = Data(data_path=data_path, dataset_name='kinetics').input_output_collocation_data(
        input_features='Time',
        output_feature='qt( catkin)',
        collocation_points=100,              
        collocation_range_multiplier=2.0     
    )
    data.processed_tensors()
    initial_qe = data.y_values.max() * 1.2

    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    model = KineticsPINNModel(input_size=1, hidden_size=20, output_size=1, initial_qe=initial_qe)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    trainining = train_kinetics_data(model=model, optimizer=optimizer, loss_fn=F.mse_loss,
                                      kinetics_data=data, device=device, num_epochs=8000)
    save_the_model(model, str(PROJECT_ROOT / 'models' / 'kinetics_pinn_model.pth'))
    print(f"Model saved successfully at {PROJECT_ROOT / 'models' / 'kinetics_pinn_model.pth'}")
