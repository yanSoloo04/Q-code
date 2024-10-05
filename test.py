import data

a = data.get_data_file('HTRU_2.csv')
b, c = data.get_samples(a, 200000, [0.0, 1.0])
print(b)