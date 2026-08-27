import numpy as np
import pandas as pd
import os

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
        return self.dataframes_dict

    def print_df(self):
        # get the name of the sheet
        print(f"Dataframe Name: {[sheet_name for sheet_name in self.dataframes_dict.keys()]}")
        print(f"Isotherm Data Shape: {self.dataframes_dict[self.sheet_name[0]].shape}")
        print(f"Kinetics Data Shape: {self.dataframes_dict[self.sheet_name[1]].shape}")
    