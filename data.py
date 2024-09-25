import pandas as pd
import random

def get_data_file():

    data_file = pd.read_csv('HTRU_2.csv')
    data_array = data_file.to_numpy()

    return data_array

def get_samples(data, sample_amount):

    sample_array = []
    zero_value = 1
    one_value = 1
    single_data = []

    while len(sample_array) < sample_amount:
        single_data = random.choice(data)
        
        if single_data[8] == 0.0 and zero_value <= (sample_amount/2):
            sample_array.append(single_data)
            zero_value += 1

        elif single_data[8] == 1.0 and one_value <= (sample_amount/2):
            sample_array.append(single_data)
            one_value += 1
        print(len(sample_array))
    return sample_array