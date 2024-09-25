import numpy as np
import pandas as pd
import random
import os

print(os.getcwd()) 


def get_data_file():

    data_file = pd.read_csv('HTRU_2.csv')
    data_array = data_file.to_numpy()

    return data_array

def get_samples(data, sample_amount):

    sample_array = []
    zero_value = 0
    one_value = 0
    single_data = []

    # for i in range(len(data)):
    #     if data[i][8] == "0" or zero_value >= (sample_amount/2):
    #         sample_array += data[i]

    #     elif data[i][8] == "1" or one_value >= (sample_amount/2):
    #         sample_array += data[i]

    while len(sample_array) <= sample_amount:
        single_data = random.choice(data)

        if single_data[8] == "0" and zero_value >= (sample_amount/2):
            sample_array += single_data
            zero_value += 1

        elif single_data[8] == "1" and one_value >= (sample_amount/2):
            sample_array += single_data
            one_value += 1

    return sample_array

#data = np.genfromtxt('HTRU_2.csv', delimiter=',', skip_header=0)

data = get_data_file()

#a = get_samples(data, 20)

#print(a)