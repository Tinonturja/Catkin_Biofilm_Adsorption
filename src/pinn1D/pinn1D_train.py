## now the toughest part
## we will get the input data, output data and collocation points for both the datasets
# Create the model for the isotherm dataset and the kinetics dataset
import pandas as pd
import os
import torch
from data import isotherm_data, kinetics_data
from pinn1D import KineticsPINNModel, IsothermPINNModel
import torch.nn.functional as F
from utility import predict_qe
import numpy as np

# declare the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_isotherm_data(model,
                        optimizer,loss_fn,
                        isotherm_data,
                        device = device,
                        num_epochs=5000):
    results_dict = {'Data Loss': [], 'Collocation Loss': [], 'Total Loss': []}
    # model into training mode
    model.train()
    model.to(device)
    # Get the input and output data for isotherm and kinetics datasets
    isotherm_input = isotherm_data.X_scaled_tensor 
    isotherm_output = isotherm_data.y_tensor
    isotherm_collocation = isotherm_data.collocated_scaled_tensor
    isotherm_input = isotherm_input.to(device)
    isotherm_output = isotherm_output.to(device)
    isotherm_collocation = isotherm_collocation.to(device)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass for isotherm data
        isotherm_pred = model(isotherm_input)
        isotherm_loss = loss_fn(isotherm_pred, isotherm_output)
        # Do the prediction on the collocation points
        qe_collocation_pred = model(isotherm_collocation)
        ## get the learnable parameters
        k_f = F.softplus(model.k_f)
        inv_n = F.softplus(model.inv_n)
        calculated_qe = predict_qe(Co = torch.tensor(isotherm_data.collocation_input, dtype=torch.float32).view(-1,1).to(device), # prediction on raw data
                                    k_f = k_f,
                                    inv_n = inv_n)
        iso_loss = torch.mean((qe_collocation_pred - calculated_qe) ** 2)
        total_loss = isotherm_loss + iso_loss
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Isotherm Training:\nEpoch [{epoch + 1}/{num_epochs}] | Data Loss: {isotherm_loss:.4f} | Collocation Loss: {iso_loss:.4f} | Total Loss: {total_loss:.4f}')
            results_dict['Data Loss'].append(isotherm_loss.item())
            results_dict['Collocation Loss'].append(iso_loss.item())
            results_dict['Total Loss'].append(total_loss.item())
    return results_dict

def train_kinetics_data(model,
                        optimizer,
                        loss_fn,
                        kinetics_data,
                        device = device,
                        num_epochs = 5000):
    results_dict = {'Data Loss': [], 'PSO Loss': [], 'Initial Loss': [], 'Total Loss': []}
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
        total_loss = kinetics_loss + pso_loss + initial_loss
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Kinetics Training:\nEpoch [{epoch + 1}/{num_epochs}] | Data Loss: {kinetics_loss.item():.4f} | PSO Loss: {pso_loss.item():.4f} | Initial Loss: {initial_loss.item():.4f} | Total Loss: {total_loss.item():.4f}')
            results_dict['Data Loss'].append(kinetics_loss.item())
            results_dict['PSO Loss'].append(pso_loss.item())
            results_dict['Initial Loss'].append(initial_loss.item())
            results_dict['Total Loss'].append(total_loss.item())
    return results_dict
torch.manual_seed(42)
torch.mps.manual_seed(42)
kinetics_model = KineticsPINNModel(input_size = 1, hidden_size = 32, output_size = 1)
isotherm_model = IsothermPINNModel(input_size = 1, hidden_size = 32, output_size = 1)
kinetics_optimizer = torch.optim.Adam(kinetics_model.parameters(), lr=0.001)
isotherm_optimizer = torch.optim.Adam(isotherm_model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
kinetics_training = train_kinetics_data(model = kinetics_model,
                                        optimizer = kinetics_optimizer,
                                        loss_fn = loss_fn,
                                        kinetics_data = kinetics_data,
                                        device = device,
                                        num_epochs = 5000)
isotherm_training = train_isotherm_data(model = isotherm_model,                                        optimizer = isotherm_optimizer,
                                        loss_fn = loss_fn,
                                        isotherm_data = isotherm_data,
                                        device = device,
                                        num_epochs = 5000)

# save the model parameters and the training results dict
# create a directory to save the model parameters and the training results

if not os.path.exists("./results"):
    os.makedirs("./results")
torch.save(kinetics_model.state_dict(), "./results/kinetics_model.pth")
torch.save(isotherm_model.state_dict(), "./results/isotherm_model.pth")
kinetics_training_df = pd.DataFrame(kinetics_training)
isotherm_training_df = pd.DataFrame(isotherm_training)
kinetics_training_df.to_csv("./results/kinetics_training_results.csv", index=False)
isotherm_training_df.to_csv("./results/isotherm_training_results.csv", index=False)
