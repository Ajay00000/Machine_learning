#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:36:18 2020

@author: aj
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('/home/aj/Downloads/50_Startups.csv')
label = LabelEncoder()

df.iloc[:,3:4] = label.fit_transform(df.iloc[:,3:4])

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

x = df.drop('Profit',axis=1)

calc_vif(x)