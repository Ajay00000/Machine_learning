#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:48:56 2020

@author: aj
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('/home/aj/Desktop/MachineLearning/Polynominal_regression/Position_Salaries.csv')
data.head()

x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

plt.scatter(x,y)
plt.show()


lin_reg=LinearRegression()
lin_reg.fit(x,y)

pr=PolynomialFeatures(degree=4)
x_poly=pr.fit_transform(x)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

# =============================================================================
# Liniear Regression 
# =============================================================================
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# =============================================================================
# Polynomial Liniear Regression
# =============================================================================
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(pr.fit_transform(x)),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()