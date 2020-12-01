#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:06:05 2020

@author: aj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('/home/aj/Desktop/MachineLearning/Multiple_Liniear_Regression/50_Startups.csv')
x = data.iloc[:,:4].values
y = data.iloc[:,-1].values

# =============================================================================
# Dummy Variables 
# =============================================================================
label = LabelEncoder()
x[:,3] = label.fit_transform(x[:,3])
transform = ColumnTransformer([('onehot',OneHotEncoder(),[3])],remainder='passthrough')
x = transform.fit_transform(x)

# Avoiding DummyVariable trap
x = x[:,1:]

# =============================================================================
# Spliting in train test  
# =============================================================================
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

# =============================================================================
#    Fitting The Model
# =============================================================================
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

# =============================================================================
#  Start with Backward Elimination
# =============================================================================
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int),values = x,axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
x_opt = x_opt.astype('float64')
regressor = sm.OLS(endog = y,exog=x_opt).fit()
regressor.summary()

