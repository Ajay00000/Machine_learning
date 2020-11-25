#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:14:54 2020

@author: aj
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('sallary.csv')

x = data.iloc[:,0].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=44,test_size=0.2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))


model.predict(x_test.reshape(-1,1))

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,model.predict(x_train.reshape(-1,1)),color='blue')
plt.title(" Sallay Vs Experience ")
plt.xlabel('Years Of Experience ')
plt.ylabel("Sallay")
plt.show()