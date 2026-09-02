import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler

class Data:
    """
    Get any excel file regarding catkin dataset
    Convert them into dataframe
    
    """
    def __init__(self, data_path, dataset_name):
        self.dataset_name = dataset_name.lower()
        if os.path.exists(data_path):
            self.data = pd.read_excel(data_path, sheet_name = None)
            if self.dataset_name == 'isotherm':
                self.dataframe = self.data['catkin']
            elif self.dataset_name == 'kinetics':
                self.dataframe = self.data['catkin (pfo pso)']
            else:
                raise ValueError("Dataset name is not right")
        else:
            raise ValueError("Data path is not right")
        

    def print_df(self):
        # get the name of the sheet
       print(f"Dataframe Shape: {self.dataframe.shape}")
       print(f"Dataframe Columns: {self.dataframe.columns.tolist()}")
       
    def input_output_collocation_data(self, input_features:str, output_feature:str, collocation_points:int):
        # set the input and the output
        self.X = self.dataframe[input_features]
        self.y = self.dataframe[output_feature]
        self.X_values = self.X.values.reshape(-1,1) 
        self.y_values = self.y.values.reshape(-1,1)
        self.collocation_input = np.linspace(start =np.min(self.X_values),stop = np.max(self.X_values), num = collocation_points)
        return self        

    def processing_data(self):
        """
        Scaled the input values of both kinetics dataset, and isotherm dataset
        """ 
        self.input_scaler_fn = StandardScaler()
        self.X_scaled = self.input_scaler_fn.fit_transform(self.X_values)
        self.collocated_scaled = self.input_scaler_fn.transform(self.collocation_input.reshape(-1,1))

    def get_tensor_scaled_data(self):
        """
        Return the tensor of the scaled input values and output values
        """
        self.X_scaled_tensor = torch.tensor(self.X_scaled, dtype = torch.float32).view(-1,1)
        self.y_tensor = torch.tensor(self.y_values, dtype = torch.float32).view(-1,1)
        self.collocated_scaled_tensor = torch.tensor(self.collocated_scaled, dtype = torch.float32).view(-1,1)
        return self

    def processed_tensors(self):
        self.processing_data() # numpy scaled data
        self.get_tensor_scaled_data() # tensor of the scaled data
        return self

# create a object of the Data class

isotherm_data = Data(data_path = "/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/data of biofilm(TINON BHAI).xlsx", dataset_name = "isotherm")
kinetics_data = Data(data_path = "/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/data of biofilm(TINON BHAI).xlsx", dataset_name = "kinetics")
isotherm_data.input_output_collocation_data(input_features = "Initial concentration", output_feature = "Qe", collocation_points = 50).processed_tensors()
kinetics_data.input_output_collocation_data(input_features = "Time", output_feature = "qt( catkin)", collocation_points = 50).processed_tensors()
print(f"Isotherm data collocation points: {isotherm_data.collocated_scaled_tensor.shape}")
print(f"Kinetics data collocation points: {kinetics_data.collocated_scaled_tensor.shape}")
print(f"Isotherm data input:\n {isotherm_data.X_scaled_tensor}")
print(f"Kinetics data input:\n {kinetics_data.X_scaled_tensor}")