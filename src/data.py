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
            self.data = pd.read_excel(data_path)

        else:
            raise ValueError("Data path is not right")