import pandas as pd
import csv
import math
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import statistics
import io
import base64
from IPython.display import display, Markdown
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

df = pd.read_csv('roof.csv')
print(df)

sizes_i = [0.3, 0.5, 1, 2, 5, 10]
n_i = ['0.3um', '0.5um', '1.0um', '2.0um', '5.0um', '10.0um']
i = [1,2,3,4,5]


def size_distr_regression_1_sqrt(sizes, nParticles) -> list[float]:

    nParticles = [val if val > 0 else 1e-10 for val in nParticles]
    y = [1.0 / math.sqrt(val) for val in nParticles]
    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, y)
    res = [slope, intercept, r_value, p_value, std_err]
    return res

a_values = []
b_values = []
r_squared_values = []
y_predicted_values = []
q_values = []

for index, row in df.iterrows():
    ni_values = row[n_i].values.astype(float)
    ni_values = np.array([val if val > 0 else 1e-10 for val in ni_values])
    yi_values = 1 / np.sqrt(ni_values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes_i, yi_values)

    a_values.append(slope)
    b_values.append(intercept)
    r_squared_values.append(r_value**2)
    y_predicted = [slope * x + intercept for x in sizes_i]
    y_predicted_values.append(y_predicted)

df['a'] = a_values
df['b'] = b_values
df['r_squared'] = r_squared_values
df['y_predicted'] = y_predicted_values

print(df[['a', 'b', 'r_squared','y_predicted']])