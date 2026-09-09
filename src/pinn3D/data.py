# get the data, and join them and create a compact dataset
import os
import numpy as np
import pandas as pd
import torch

class CombinedData:
    def __init__(self, data_path):
        """
        Data path, location of a excel file, which has multi
        
        """
        self.df_list = []
        if os.path.exists(data_path):
            self.data_sheets_dict = pd.read_excel(data_path, sheet_name = None)
        else:
            raise ValueError("Data path is not right")

    def process_kinetics_data(self):
        dataset_name = 'catkin (pfo pso)'
        self.kinetics_data_df = self.data_sheets_dict[dataset_name]
        self.kinetics_data_df['Concentration'] = 40
        self.kinetics_data_df['Dosage'] = 24
        self.kinetics_data_df.rename(columns = {'qt( catkin)': 'Adsorption' }, inplace = True)
        self.kinetics_data_df = self.kinetics_data_df[['Time', 'Concentration', 'Dosage', 'Adsorption']]
        return self.kinetics_data_df
    def process_isotherm_data(self):
        dataset_name = 'catkin'        
        self.isotherm_data_df = self.data_sheets_dict[dataset_name]
        self.isotherm_data_df['Time'] = 170
        self.isotherm_data_df['Dosage'] = 10
        #self.isotherm_data_df.drop(columns = ['LN CE','LN QE', 'Co-Ce', 'Ce/Qe', 'removal percentage(catkin)', 'absorbance','Ce'], inplace = True)
        self.isotherm_data_df.rename(columns={'Initial concentration':'Concentration', 'Qe':'Adsorption'}, inplace=True)
        self.isotherm_data_df = self.isotherm_data_df[['Time', 'Concentration', 'Dosage', 'Adsorption']]
        return self.isotherm_data_df
data = CombinedData(data_path = "/Users/tinonturjamajumder/Catkin_Biofilm_Adsorption/data of biofilm(TINON BHAI).xlsx")
kinetics_df = data.process_kinetics_data()
isotherm_df = data.process_isotherm_data()
print(kinetics_df)
print(isotherm_df)