# coding:utf-8
'''
@time:    Created on  2018-03-16 10:12:05
@author:  Lanqing
@Func:    Plot some points
'''

import numpy
import matplotlib.pyplot as plt
import pandas as pd

def fetch_X_Y_Z_pair(string_):
    b = string_.split(';')
    X, Y, Z = [], [], []
    for item in b:
        item = item.split(',')
        if item and  len(item) > 2 :
            # print(item)
            x, y, z = item[0], item[1], item[2]
            X.append(int(x))
            Y.append(int(y))
            Z.append(int(z))
    print(X, Y)
    return X, Y, Z

# Initial and define variables
fid = open('dsjtzs_txfz_training.txt', 'r')
data_dict = {}
data_pandas = pd.DataFrame()
N = 5  # sample number

# read data
data = pd.read_csv(fid, delimiter=' ', header=None)
data.columns = ['Id', 'Points', 'Target', 'Label']

# Statistics info
print(len(data[data['Label'] == 0]))
print(len(data[data['Label'] == 1]))

# fetch samples Machine points 
machine_points = data[data['Label'] == 0]
human_points = data[data['Label'] == 1]
samples_machine = machine_points['Points'][:N].values
samples_humans = human_points['Points'][:N].values

# plot
X, Y = [], []
for (item1, item2) in zip(samples_machine, samples_humans):
    X_Mch, Y_Mch, T_Mch = fetch_X_Y_Z_pair(item1)
    X_Hum, Y_Hum, T_Hum = fetch_X_Y_Z_pair(item2)
    plt.style.use('fivethirtyeight')
    # Plot X and Y
    plt.plot(X_Mch, Y_Mch, '*', X_Hum, Y_Hum, '-')
    plt.xlabel('X axis')  
    plt.ylabel('Y axis')  
    plt.legend(['Machine', 'Human'], loc='upper left', fancybox=True, shadow=True)
    plt.show()

