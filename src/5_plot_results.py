# coding:utf-8
'''
@time:    Created on  2018-06-11 19:19:14
@author:  Lanqing
@Func:    
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('../data/result_final.csv')
data['Gini'] = 1 / data['Gini']

tt = data.T
tt.columns = data['Classifier']
tt.columns = tt.columns
tt = tt.iloc[1:, :]

data = tt
print(data.describe())

gini = data.iloc[0, :].T
data = data.iloc[1:, :]

#### 绘制第一部分
data.plot(figsize=(24, 12), kind='bar')
plt.legend(loc='best')

yDown, yUp = 0.88, 1.0
plt.title('Performance of different models', fontsize=30)  
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.axis([-0.5, 4, yDown, yUp])  # 0.240 0.2455 50ms #0.2375, 0.245 100ms
plt.xlabel("Models", size=24)
plt.ylabel("Performance", size=24)  
plt.savefig('../data/' + 'results.pdf')
plt.show()

#### 绘制第二部分
gini.plot(figsize=(24, 12), kind='bar')
plt.legend(loc='best')

plt.title('Performance of different models', fontsize=30)  
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Models", size=24)
plt.ylabel("Performance", size=24)  
plt.savefig('../data/' + 'gini.pdf')
plt.show()
