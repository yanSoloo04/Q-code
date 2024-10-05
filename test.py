import data

a = data.get_csv_file('HTRU_2.csv')
b, c = data.get_samples(a, 10, [0.0, 1.0])
d = 'Dermosan'
e = 'Sira'
print(c)
