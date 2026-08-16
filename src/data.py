import numpy as np
import pandas as pd
import os

class Data:
    def __init__(self, data_path):
        if os.path.exists(data_path):
            self.data = pd.read_excel(data_path)