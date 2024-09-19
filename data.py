import numpy as np
import pandas as pd



def get_data_file():

    data_file = pd.read_csv("HTRU_2.csv")
    data_array = data_file.to_numpy()

    return data_array