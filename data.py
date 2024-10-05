import pandas as pd
import random
from numpy.typing import NDArray
import numpy as np

def get_csv_file(file_path : str) -> NDArray:
    """
    Get values in a csv file

    Parameters: 
        file : The path to a file with the data set to use the algorythm with 

    Returns:
        numpy array with the data set
    """

    data_file = pd.read_csv(file_path)
    data_array = data_file.to_numpy()

    return data_array

def get_xlsx_file(file_path : str) -> NDArray:
    """
    Get values in a xlsx (excel) file

    Parameters: 
        file : The path to a file with the data set to use the algorythm with 

    Returns:
        numpy array with the data set
    """

    data_file = pd.read_excel(file_path)
    data_array = data_file.to_numpy()

    return data_array

def get_samples(data : NDArray, sample_amount : int, label_values : NDArray) -> tuple[NDArray, NDArray]:

    """
    Makes an array with a balanced amount of data type

    Args: 
        data : An array with the data 
        sample_amount : The amount of sample to try the algorithm with

    Returns:
        Sample_array : A numpy array with a balenced set of data 
        Index_array : A numpy array with the labels attached to the data
    """

    sample_array = []
    index_array = []

    label_value1 = list([single_data1 for single_data1 in data if single_data1[-1] == label_values[0]])
    label_value2 = list([single_data2 for single_data2 in data if single_data2[-1] == label_values[1]])

    sample_amount = min(sample_amount, len(label_value1), len(label_value2))

    random_label_value1 = random.sample(label_value1, int(sample_amount/2))
    random_label_value2 = random.sample(label_value2, int(sample_amount/2))

    sample_array = random_label_value2 + random_label_value1
    random.shuffle(sample_array)

    for index in sample_array :
        if index[-1] == label_values[0] :
            index_array.append(-1)
        else :
            index_array.append(1)
    
    return np.array(sample_array), np.array(index_array)