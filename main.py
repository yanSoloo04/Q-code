import data
import MLPC
import QCNN
import QSVM
import VQC

data_set = data.get_csv_file('HTRU_2.csv')

sample_array = data.get_samples(data_set, 100, [0,1])

