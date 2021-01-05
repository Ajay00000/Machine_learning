import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('Position_Salaries.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2].values
# Scaling the X values
scX = StandardScaler()
scY = StandardScaler()
x = scX.fit_transform(x)
y = scY.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')

regressor.fit(x,y)
scY.inverse_transform(regressor.predict(scX.transform([[6.5]])))

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('level VS experience (SVR)')
plt.xlabel('Level')
plt.ylabel('Experience')
plt.show()