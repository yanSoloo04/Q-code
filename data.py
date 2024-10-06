import pandas as pd
import random
import numpy as np

def get_data_file():

    data_file = pd.read_csv('HTRU_2.csv')
    data_array = data_file.to_numpy()

    return data_array

def get_samples(data, sample_amount):

    sample_array = []
    index_array = []
    zero_value = 1
    one_value = 1
    single_data = []

    while len(sample_array) < sample_amount:
        single_data = random.choice(data)
        
        if single_data[8] == 0 and zero_value <= (sample_amount/2):
            single_index = 0
            single_data = single_data[:8]
            sample_array.append(single_data)
            index_array.append(single_index)
            zero_value += 1

        elif single_data[8] == 1 and one_value <= (sample_amount/2):
            single_index = 1
            single_data = single_data[:8]
            sample_array.append(single_data)
            index_array.append(single_index)
            one_value += 1
    
    return np.array(sample_array), np.array(index_array)