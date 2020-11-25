#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:00:27 2020

@author: aj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Data.csv')

x = data.iloc[:,:3].values
y = data.iloc[:,-1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

x[:,1:3] = imputer.fit_transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
label = LabelEncoder()
x[:,0] = label.fit_transform(x[:,0])

transform = ColumnTransformer(transformers=[('onehot',OneHotEncoder(),[0])],remainder='passthrough')
x = transform.fit_transform(x)

y = label.fit_transform(y)

# =============================================================================
# Spliting data into Training set and Test set
# =============================================================================

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=44,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
