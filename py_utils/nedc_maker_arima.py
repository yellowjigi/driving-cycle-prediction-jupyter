# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 07:44:32 2020

@author: MNI
"""
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

import numpy as np

import matplotlib.pyplot as plt

def nedc_maker(t):
    
    offset = 0.0
    a = 0
    v = 0
    
    if t > 1180:
        t = t - 1180
    
    if t < 195:
        offset = 0.0
    elif t >= 195 and t < 390:
        offset = 195
    elif t > 390 and t < 585:
        offset = 390
    elif t >= 585 and t < 780:
        offset = 585
    
    if t >= (0 + offset) and t < (11 + offset):
        a = 0
        v = 0
    elif t >= (11 + offset) and t < (15 + offset):
        a = 1.04
        v = a * (t - (11 + offset))
    elif t >= (15 + offset) and t <= (23 + offset):
        a = 0
        v = 4.16
    elif t > (23 + offset) and t < (28 + offset):
        a = -0.83
        v = 4.16 + a * (t - (23 + offset))
    elif t >= (28 + offset) and t <= (49+ offset):
        a=0
        v=0
    elif t > (49+offset) and t < (55 + offset):
        a = 0.69
        v = a*(t - (49 + offset))
    elif t >= (55+ offset) and t < (61+ offset):
        a = 0.79
        v = 4.16 + a * (t - (55 + offset))
    elif t >= (61 + offset) and t <= (85 + offset):
        a = 0
        v = 8.9
    elif t > (85 + offset) and t < (96 + offset):
        a = -0.81
        v = 8.9 + a * (t - (85 + offset))
    elif t >= (96 + offset) and t <= (117 + offset):
        a = 0
        v = 0
    elif t > (117 + offset) and t < (123 + offset):
        a = 0.69
        v = a * (t - (117 + offset))
    elif t >= (123 + offset) and t < (134 + offset):
        a = 0.51
        v = 4.16 + a * (t - (123 + offset))
    elif t >= (134 + offset) and t < (143 + offset):
        a = 0.46
        v = 9.72 + a * ( t - (134 + offset))
    elif t >= (143 + offset)  and t <= (155 + offset):
        a = 0
        v = 13.88
    elif t > (155 + offset) and t < (163 + offset):
        a = -0.52
        v = 13.88 + a * (t - (155 + offset))
    elif t >= (163 + offset) and t <= (178 + offset):
        a = 0
        v = 9.72
    elif t > (178 + offset) and t < (188 + offset):
        a = -0.97
        v = 9.72 + a * (t - (178 + offset))
    elif t >= (188 + offset) and t <= (195 + offset):
        a = 0
        v = 0
    elif t >= 780 and t <= 800:
        a = 0
        v = 0
    elif t > 800 and t < 806:
        a = 0.69
        v = a * (t - 800)
    elif t >= 806 and t < 817:
        a = 0.51
        v = 4.16 + a * (t - 806)
    elif t >= 817 and t < 827:
        a = 0.42
        v = 9.72 + a * (t - 817)
    elif t >= 827 and t < 841:
        a = 0.4
        v = 13.88 + a * (t - 827)
    elif t >= 841 and t <= 891:
        a = 0
        v = 19.44
    elif t > 891 and t < 899:
        a = -0.69
        v = 19.44 + a * (t - 891)
    elif t >= 899 and t <= 968:
        a = 0
        v = 13.88
    elif t > 968 and t < 981:
        a = 0.43
        v = 13.88 + a * (t - 968)
    elif t >= 981 and t < 1031:
        a = 0
        v = 19.44
    elif t >= 1031 and t < 1066:
        a = 0.24
        v = 19.44 +  a * (t - 1031)
    elif t >= 1066 and t < 1096:
        a = 0
        v = 27.77
    elif t >= 1096 and t < 1116:
        a = 0.28
        v = 27.77 + a * (t - 1096)
    elif t >= 1116 and t <= 1126:
        a = 0
        v = 33.33
    elif t > 1126 and t < 1142:
        a = -0.69
        v = 33.33 + a * (t - 1126)   
    elif t >= 1142 and t < 1150:
        a = -1.04
        v = 22.22 + a * (t - 1142)
    elif t >= 1150 and t < 1160:
        a = -1.39
        v = 13.88 + a * (t - 1150)    
    elif t >= 1160 and t < 1180:
        a = 0
        v = 0
    
    return v


# for i in range(195): 
#     print(i, nedc_maker(i))

# x1 = np.linspace(0,2360,2361)
# y1 = np.zeros(x1.shape[0])

# for i in range(2360):
#     y1[i] = nedc_maker(x1[i])

# print(y1)

# plt.plot(x1, y1)
    
x = np.linspace(0,585,586)
y = np.zeros(x.shape[0])

for i in range(585):
    y[i] = nedc_maker(x[i])

print(y)

#plt.plot(x,y)

# def parser(x):
#     return datetime.strptime('190'+x, '%Y-%m')

# series = read_csv('shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# series.index = series.index.to_period('M')

X = y
# X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' %(yhat, obs))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE : %.3f' % rmse)

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()    