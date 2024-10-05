import data

a = data.get_csv_file('HTRU_2.csv')
b, c = data.get_samples(a, 10, [0.0, 1.0])
d = ['DERMASON', 'SIRA']
e = data.get_xlsx_file('Dry_Bean_Dataset.xlsx')
f, g = data.get_samples(e, 10, d)
print(f)
