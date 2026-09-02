## now the toughest part
## we will get the input data, output data and collocation points for both the datasets
# Create the model for the isotherm dataset and the kinetics dataset
import torch
from data import isotherm_data, kinetics_data
from pinn1D import PINN1DModel
import torch.nn.functional as F

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# Define the model
model = PINN1DModel(input_size=1, hidden_size=20, output_size=1).to(device)
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train_pinn1D(model, optimizer,loss_fn, isotherm_data, kinetics_data = None, device = device ,num_epochs=1000):
    # model into training mode
    model.train()
    # Get the input and output data for isotherm and kinetics datasets
    if kinetics_data is None:
        isotherm_input = isotherm_data.X_scaled_tensor
        isotherm_output = isotherm_data.y_tensor
        isotherm_collocation = isotherm_data.collocated_scaled_tensor
        isotherm_input = isotherm_input.to(device)
        isotherm_output = isotherm_output.to(device)
        isotherm_collocation = isotherm_collocation.to(device)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Forward pass for isotherm data
            isotherm_pred = model(isotherm_data.X_scaled_tensor)
            isotherm_loss = loss_fn(isotherm_pred, isotherm_output)
            # Do the prediction on the collocation points
            collocation_pred = model(isotherm_collocation)
            ## get the learnable parameters
            k_f = F.softplus(model.k_f)
            inv_n = F.softplus(model.inv_n)
            


            
            # Backward pass and optimization
            isotherm_loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {isotherm_loss.item():.4f}')
    else:
        
        kinetics_input = kinetics_data.X_scaled_tensor
        kinetics_output = kinetics_data.y_tensor
        kinetics_collocation = kinetics_data.collocated_scaled_tensor
    # pass the input and output data to the device
        kinetics_input = kinetics_input.to(device)
        kinetics_output = kinetics_output.to(device)
        kinetics_collocation = kinetics_collocation.to(device)
    
    # Training loop

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass for isotherm data
        isotherm_pred = model(isotherm_data.X_scaled_tensor)
        isotherm_loss = F.mse_loss(isotherm_pred, isotherm_data.y_tensor)
        
        # Forward pass for kinetics data
        kinetics_pred = model(kinetics_data.X_scaled_tensor)
        kinetics_loss = F.mse_loss(kinetics_pred, kinetics_data.y_tensor)
        
        # Total loss
        total_loss = isotherm_loss + kinetics_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')
