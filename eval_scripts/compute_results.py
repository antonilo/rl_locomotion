import numpy as np
import pandas as pd
import sys
import copy
from prettytable import PrettyTable

path = sys.argv[1]
max_steps = 1100
dividers = ['forward_r', 'energy', 'smoothness', 'ground_impact']

df = pd.read_csv(path)
num_steps = copy.deepcopy(df['num_steps'].values)
df['num_steps'] /= max_steps
df['success_rate'] = df['num_steps'] > 0.99

print("Num experiments is {}".format(df.shape[0]))

names = df.columns.tolist()
medians = []

for n in names:
    if n in dividers:
        df[n] /= num_steps
    value = np.mean(df[n].values)
    medians.append(value)

t = PrettyTable(names)
t.add_row(medians)
print(t)






