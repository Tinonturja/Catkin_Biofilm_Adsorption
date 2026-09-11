# get the data, and join them and create a compact dataset
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    kinetics_sheet: str = 'catkin (pfo pso)'
    isotherm_sheet: str = 'catkin'

    kinetics_concentration: float = 40
    kinetics_dosage: float = 1/24

    isotherm_time: float = 170
    isotherm_dosage: float = 1/10

    input_columns: tuple[str, ...] = (
        'Time',
        'Concentration',
        'Dosage'
    )
    target_column: str = 'Adsorption'
    time_collocation_num:int = 25
    conc_collocation_num:int = 25
    vm_collocation_num:int = 4
    biofilm_mass:int = 5

class CombinedData:
    def __init__(self, data_path:str, config: DataConfig):
        """
        Data path, location of a excel file, which has multi
        
        """
        self.data_path = Path(data_path)
        self.config = config
        self.df_list = []
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}"
            )
        self.data_sheets_dict = pd.read_excel(self.data_path, sheet_name = None)
        
    def process_kinetics_data(self) -> pd.DataFrame:
        df = self.data_sheets_dict[
            self.config.kinetics_sheet
        ].copy()
        df['Concentration'] = (self.config.kinetics_concentration)
        df['Dosage'] = (self.config.kinetics_dosage)
        df = df.rename(columns = {'qt( catkin)': 'Adsorption' })
        self.kinetics_data_df = df[['Time', 'Concentration', 'Dosage', 'Adsorption']]
        return self.kinetics_data_df
    
    def process_isotherm_data(self)->pd.DataFrame:
        df = self.data_sheets_dict[
            self.config.isotherm_sheet
        ].copy()

        df["Time"] = self.config.isotherm_time

        df["Dosage"] = self.config.isotherm_dosage

        df = df.rename(
            columns={
                "Initial concentration": "Concentration",
                "Qe": "Adsorption",
            }
        )

        self.isotherm_data_df = df[
            [
                "Time",
                "Concentration",
                "Dosage",
                "Adsorption",
            ]
        ]

        return self.isotherm_data_df

    def combined_data(self)->pd.DataFrame:
        self.combined_df = pd.concat([self.kinetics_data_df, self.isotherm_data_df],
                                     ignore_index=True)
        
        return self.combined_df
    
    def get_input_output(self):
        self.input = self.combined_df[list(self.config.input_columns)]
        self.output = self.combined_df[self.config.target_column]
        # turn them into tensor
        self.input_tensor = torch.tensor(self.input.to_numpy(), dtype = torch.float32)
        self.output_tensor = torch.tensor(self.output.to_numpy(), dtype = torch.float32)
        return self
    
    def collocation_data(self):
        time_alloc = np.linspace(0, 200, num = self.config.time_collocation_num)
        vm_alloc = np.linspace(0.030, 0.120, num = self.config.vm_collocation_num)
        conc_alloc = np.linspace(15, 65, num = self.config.conc_collocation_num)
        T,C,V = np.meshgrid(time_alloc,conc_alloc,vm_alloc)
        self.allocated_input = np.column_stack([T.reshape(-1,1),
        C.reshape(-1,1),
        V.reshape(-1,1)])
        self.collocated_input_tensor = torch.tensor(self.allocated_input,
        requires_grad=True)
        return self
    
    def scaling_data(self):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        scaled_input = self.input_scaler.fit_transform(self.input)
        scaled_output = self.output_scaler.fit_transform(self.output.to_numpy().reshape(-1,1))
        self.scaled_input_tensor = torch.tensor(scaled_input, dtype = torch.float32)
        self.scaled_output_tensor = torch.tensor(scaled_output, dtype = torch.float32)
        self.input_mean = self.input_scaler.mean_
        self.input_std = self.input_scaler.scale_
        self.output_mean = self.output_scaler.mean_
        self.output_std = self.output_scaler.scale_
        self.input_stats = {
            feature: {
                "mean": mean,
                "std": std
            }
            for feature, mean, std in zip(
                self.config.input_columns,
                self.input_mean,
                self.input_std
            )
        } 
        return self
    
    def whole_data_processing(self):
        self.process_kinetics_data()
        self.process_isotherm_data()
        self.combined_data()
        self.get_input_output()
        self.scaling_data()
        return self
    
def load_data(data_path):
    config = DataConfig()

    data = CombinedData(
        data_path=data_path,
        config=config
    )

    data.whole_data_processing()

    return data
