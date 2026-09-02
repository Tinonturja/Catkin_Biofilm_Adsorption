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
    def __init__(self, data_path):
        if os.path.exists(data_path):
            self.data = pd.read_excel(data_path, sheet_name = None)
        else:
            raise ValueError("Data path is not right")

    def get_dataframe(self):
        self.dataframes_dict = {}
        for sheet, df in self.data.items():
            if 'catkin' in sheet:
                self.dataframes_dict[sheet] = df
        self.sheet_name = [sheet_name for sheet_name in self.dataframes_dict.keys()]
        # got the two dataset
        self.isotherm_df = self.dataframes_dict[self.sheet_name[0]]
        self.kinetics_df = self.dataframes_dict[self.sheet_name[1]]
        return self

    def print_df(self):
        # get the name of the sheet
        print(f"Dataframe Name: {[sheet_name for sheet_name in self.dataframes_dict.keys()]}")
        print(f"Isotherm Data Shape: {self.dataframes_dict[self.sheet_name[0]].shape}")
        print(f"Kinetics Data Shape: {self.dataframes_dict[self.sheet_name[1]].shape}")

    def get_inputoutputcollocation_data(self,input_kinetic_features:str, input_isotherm_features:str, output_kinetics_feature:str, output_isotherm_feature:str):
        # set the input and the output
        """
        Get both input and output data for the kinetics and isotherm dataset
        Input, output data, and their numpy values are stored in the class attributes
        """
        self.X_kin = self.kinetics_df[input_kinetic_features]
        self.X_iso = self.isotherm_df[input_isotherm_features]
        self.y_kin_output = self.kinetics_df[output_kinetics_feature]
        self.y_iso_output = self.isotherm_df[output_isotherm_feature]
        self.X_kin_values = self.X_kin.values
        self.X_iso_values = self.X_iso.values
        self.output_kinetics_values = self.y_kin_output.values
        self.output_isotherm_values = self.y_iso_output.values
        return self

    def processing_data(self):
        """
        Scaled the input values of both kinetics dataset, and isotherm dataset
        """
        self.input_kinetics_scaler_fn = StandardScaler()
        self.input_isotherm_scaler_fn = StandardScaler()
        self.scaled_kinetics_input_data = self.input_kinetics_scaler_fn.fit_transform(self.X_kin_values.reshape(-1,1))
        self.scaled_isotherm_input_data = self.input_isotherm_scaler_fn.fit_transform(self.X_iso_values.reshape(-1,1))

    def get_tensor_scaled_data(self):
        """
        Return the tensor of the scaled input values and output values
        """
        self.scaled_kinetics_input_tensor = torch.tensor(self.scaled_kinetics_input_data, dtype=torch.float32).view(-1,1)
        self.scaled_isotherm_input_tensor = torch.tensor(self.scaled_isotherm_input_data, dtype=torch.float32).view(-1,1) 
        self.output_kinetics_tensor = torch.tensor(self.output_kinetics_values, dtype=torch.float32).view(-1,1)
        self.output_isotherm_tensor = torch.tensor(self.output_isotherm_values, dtype=torch.float32).view(-1,1)
        return self
    
    def get_collocation_point(self, num_collocation_points:int):
        """
        Create the scaled collocation point for both kinetics and isotherm dataset
        For kinetics dataset, 
        Collocation points are created from the minimum and maximum of the input time values
        For isotherm dataset, collocation points are created from the minimum and maximum of the input concentration values
        """
        self.num_collocation_points = num_collocation_points
        self.collocation_kinetics_input = np.linspace(start =np.min(self.X_kin),stop = np.max(self.X_kin), num = num_collocation_points)
        self.collocation_isotherm_input = np.linspace(start =np.min(self.X_iso),stop = np.max(self.X_iso), num = num_collocation_points)
        self.scaled_collocation_kinetics_input = torch.tensor(self.input_kinetics_scaler_fn.transform(self.collocation_kinetics_input.reshape(-1,1)), dtype = torch.float32).view(-1,1)
        self.scaled_collocation_isotherm_input = torch.tensor(self.input_isotherm_scaler_fn.transform(self.collocation_isotherm_input.reshape(-1,1)), dtype = torch.float32).view(-1,1)                                                                                                                                                                                                                                                                                                                                    
        return self

    def processed_tensors(self, collocation_point:int):
        self.processing_data() # numpy scaled data
        self.get_tensor_scaled_data() # tensor of the scaled data
        self.get_collocation_point(num_collocation_points=collocation_point)
        return self

data = Data(data_path = "./data of biofilm(TINON BHAI).xlsx")
print(f"Isotherm Features: {data.get_dataframe().isotherm_df.columns}")
print(f"Kinetics Features: {data.get_dataframe().kinetics_df.columns}")
data.get_inputoutputcollocation_data(input_isotherm_features='Initial concentration',
                                     input_kinetic_features= 'Time',
                                     output_kinetics_feature='qt( catkin)',
                                     output_isotherm_feature='Qe')
print(data.X_kin_values)
# check the collocation point
data.processed_tensors(collocation_point=10)
print(data.scaled_collocation_kinetics_input)
print(data.scaled_collocation_isotherm_input)